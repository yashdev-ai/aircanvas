"""
Air Canvas v3 - High Accuracy + Custom Colors
==============================================
GESTURES:
  1 finger  = Draw / interact
  2 fingers = Lift pen (move freely)
  5 fingers = Clear canvas

KEYBOARD:
  Z = Undo        Y = Redo
  C = Clear       E = Toggle eraser
  M = Cycle mode (Draw / Rectangle / Circle / Line / Text)
  T = Enter text  (then type, Enter to stamp)
  Q / ESC = Quit
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from collections import deque
import urllib.request, os

# ── Model download ─────────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model (~9MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH)
    print("Done.")

# ── Palette ────────────────────────────────────────────────────────────────────
DEFAULT_COLORS = [
    ("Pink",   (203,  65, 227)),
    ("Blue",   (235,  85,  20)),
    ("Green",  ( 30, 220,  30)),
    ("White",  (240, 240, 240)),
    ("Yellow", ( 20, 215, 235)),
    ("Red",    ( 20,  20, 230)),
    ("Cyan",   (220, 195,  10)),
    ("Orange", ( 20, 140, 255)),
]
CUSTOM_SLOT = ("Custom", (255, 255, 255))   # mutable
ERASER_COLOR = (0, 0, 0)
TOOLBAR_H    = 150
MODES        = ["DRAW", "RECT", "CIRCLE", "LINE", "TEXT"]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

# ── One Euro Filter (much better than moving average for drawing) ──────────────
class OneEuroFilter:
    """Adaptive low-pass filter — reduces lag at fast movements, jitter at slow."""
    def __init__(self, freq=30, mincutoff=1.5, beta=0.08, dcutoff=1.0):
        self.freq      = freq
        self.mincutoff = mincutoff
        self.beta      = beta
        self.dcutoff   = dcutoff
        self.x_prev    = None
        self.dx_prev   = 0.0
        self.t_prev    = None

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx   = (x - self.x_prev) * self.freq
        edx  = self.dx_prev + self._alpha(self.dcutoff) * (dx - self.dx_prev)
        cutoff = self.mincutoff + self.beta * abs(edx)
        result = self.x_prev + self._alpha(cutoff) * (x - self.x_prev)
        self.x_prev = result
        self.dx_prev = edx
        return result

    def reset(self):
        self.x_prev  = None
        self.dx_prev = 0.0


class PointFilter:
    def __init__(self):
        self.fx = OneEuroFilter()
        self.fy = OneEuroFilter()
    def update(self, x, y):
        return int(self.fx.filter(x)), int(self.fy.filter(y))
    def reset(self):
        self.fx.reset(); self.fy.reset()


# ── Gesture stabilizer — requires N consistent frames before switching ─────────
class GestureStabilizer:
    def __init__(self, required=4):
        self.required  = required
        self.candidate = None
        self.count     = 0
        self.current   = None
    def update(self, gesture):
        if gesture == self.candidate:
            self.count += 1
        else:
            self.candidate = gesture
            self.count     = 1
        if self.count >= self.required:
            self.current = self.candidate
        return self.current


# ── Canvas history ─────────────────────────────────────────────────────────────
class History:
    def __init__(self, maxlen=25):
        self.past   = deque(maxlen=maxlen)
        self.future = deque(maxlen=maxlen)
    def push(self, canvas):
        self.past.append(canvas.copy())
        self.future.clear()
    def undo(self, canvas):
        if self.past:
            self.future.append(canvas.copy())
            return self.past.pop()
        return canvas
    def redo(self, canvas):
        if self.future:
            self.past.append(canvas.copy())
            return self.future.pop()
        return canvas


# ── Color picker wheel ─────────────────────────────────────────────────────────
def make_color_picker(size=300):
    """Generate HSV color wheel image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy, r = size//2, size//2, size//2 - 10
    for y in range(size):
        for x in range(size):
            dx, dy = x - cx, y - cy
            dist = np.sqrt(dx*dx + dy*dy)
            if dist <= r:
                angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
                sat   = dist / r
                hsv   = np.uint8([[[int(angle/2), int(sat*255), 255]]])
                bgr   = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
                img[y, x] = bgr
    # center white
    cv2.circle(img, (cx, cy), 18, (255,255,255), -1)
    cv2.circle(img, (cx, cy), 18, (180,180,180), 1)
    cv2.putText(img, "W", (cx-6, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return img


def sample_wheel(img, x, y):
    """Sample BGR color from picker at pixel (x,y)."""
    h, w = img.shape[:2]
    x = max(0, min(w-1, x))
    y = max(0, min(h-1, y))
    return tuple(int(v) for v in img[y, x])


# ── Toolbar ────────────────────────────────────────────────────────────────────
def make_toolbar(width, colors, sel_color, is_eraser, mode, brush_sz, opacity):
    bar = np.zeros((TOOLBAR_H, width, 3), dtype=np.uint8)
    bar[:] = (18, 18, 24)
    cv2.line(bar, (0, TOOLBAR_H-1), (width, TOOLBAR_H-1), (55,55,65), 1)

    icon_rects = []
    icon_w, pad = 76, 10
    x = pad

    for i, (name, bgr) in enumerate(colors):
        cx, cy = x + icon_w//2, TOOLBAR_H//2 - 10

        # Checkerboard bg for custom slot
        if name == "Custom":
            for ty in range(cy-24, cy+24, 8):
                for tx in range(cx-24, cx+24, 8):
                    shade = (60,60,60) if ((tx+ty)//8)%2==0 else (90,90,90)
                    cv2.rectangle(bar, (tx,ty), (tx+8,ty+8), shade, -1)
            # clip to circle
            mask_c = np.zeros((TOOLBAR_H, icon_w+x-x, 1), dtype=np.uint8)

        cv2.circle(bar, (cx, cy), 24, bgr, -1)
        cv2.circle(bar, (cx, cy), 24, (70,70,80), 1)

        if not is_eraser and i == sel_color:
            cv2.circle(bar, (cx, cy), 28, (255,255,255), 2)
            cv2.circle(bar, (cx, cy), 29, (0,0,0), 1)

        cv2.putText(bar, name[:6], (cx - len(name[:6])*3, cy + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (140,140,155), 1)

        icon_rects.append((x, 0, x+icon_w, TOOLBAR_H, ("color", i)))
        x += icon_w + 4

    # Custom color "+" button
    plus_x1, plus_x2 = x, x+56
    plus_y1, plus_y2 = 38, TOOLBAR_H-38
    cv2.rectangle(bar, (plus_x1,plus_y1),(plus_x2,plus_y2),(40,40,55),-1)
    cv2.rectangle(bar, (plus_x1,plus_y1),(plus_x2,plus_y2),(80,80,100),1)
    pcx = (plus_x1+plus_x2)//2
    pcy = (plus_y1+plus_y2)//2
    cv2.line(bar,(pcx-10,pcy),(pcx+10,pcy),(200,200,210),2)
    cv2.line(bar,(pcx,pcy-10),(pcx,pcy+10),(200,200,210),2)
    cv2.putText(bar,"+ CLR",(plus_x1+2,plus_y2+12),cv2.FONT_HERSHEY_SIMPLEX,0.28,(130,130,150),1)
    icon_rects.append((x, 0, x+60, TOOLBAR_H, ("picker", -1)))
    x += 66

    # Divider
    cv2.line(bar,(x,18),(x,TOOLBAR_H-18),(55,55,65),1)
    x += 8

    # Eraser
    ew = 76
    ex1,ex2 = x, x+ew
    ey1,ey2 = 32, TOOLBAR_H-32
    col = (195,190,205) if is_eraser else (55,53,65)
    cv2.rectangle(bar,(ex1,ey1),(ex2,ey2),col,-1)
    cv2.rectangle(bar,(ex1,ey1),(ex2,ey2),(90,90,105),1)
    cv2.putText(bar,"ERASE",(ex1+8,(ey1+ey2)//2+5),
                cv2.FONT_HERSHEY_SIMPLEX,0.48,
                (15,15,15) if is_eraser else (190,190,205),1)
    if is_eraser:
        cv2.rectangle(bar,(ex1-2,ey1-2),(ex2+2,ey2+2),(255,255,255),2)
    icon_rects.append((x,0,x+ew,TOOLBAR_H,("eraser",-1)))
    x += ew+8

    # Mode buttons
    for m in MODES:
        mw = 70
        mx1,mx2 = x, x+mw
        my1,my2 = 32, TOOLBAR_H-32
        active = (m == mode)
        bg = (45,110,190) if active else (35,35,48)
        cv2.rectangle(bar,(mx1,my1),(mx2,my2),bg,-1)
        cv2.rectangle(bar,(mx1,my1),(mx2,my2),(75,75,95),1)
        cv2.putText(bar,m,(mx1+4,(my1+my2)//2+5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.40,
                    (255,255,255) if active else (155,155,175),1)
        icon_rects.append((x,0,x+mw,TOOLBAR_H,("mode",m)))
        x += mw+4

    return bar, icon_rects


def draw_sliders(frame, fh, brush_sz, opacity):
    sw,sh = 44,300
    gap   = 20
    sx1   = frame.shape[1] - sw*2 - gap - 15
    sx2   = frame.shape[1] - sw - 15
    sy    = (fh-sh)//2

    def one(sx, val, lo, hi, label, col):
        cv2.rectangle(frame,(sx,sy),(sx+sw,sy+sh),(18,18,24),-1)
        cv2.rectangle(frame,(sx,sy),(sx+sw,sy+sh),(55,55,65),1)
        tx = sx+sw//2
        cv2.line(frame,(tx,sy+12),(tx,sy+sh-12),(55,55,65),2)
        ratio = (val-lo)/(hi-lo)
        ty = int(sy+sh-12-ratio*(sh-24))
        cv2.circle(frame,(tx,ty),12,col,-1)
        cv2.circle(frame,(tx,ty),12,(90,90,105),1)
        cv2.putText(frame,label,(sx+2,sy+sh+18),cv2.FONT_HERSHEY_SIMPLEX,0.38,(170,170,185),1)
        cv2.putText(frame,str(val),(sx+4,sy+sh+32),cv2.FONT_HERSHEY_SIMPLEX,0.36,(210,210,225),1)
        return sx,sy,sw,sh,lo,hi

    r1 = one(sx1, brush_sz, 3, 80, "SIZE", (195,195,210))
    r2 = one(sx2, int(opacity*100), 10, 100, "OPAC", (90,170,250))
    return r1, r2


# ── Finger detection ───────────────────────────────────────────────────────────
TIPS = [4, 8, 12, 16, 20]
PIPS = [3, 6, 10, 14, 18]

def fingers_up(lm):
    up = [1 if lm[4][0] < lm[3][0] else 0]
    for t,p in zip(TIPS[1:], PIPS[1:]):
        up.append(1 if lm[t][1] < lm[p][1] else 0)
    return up

def gesture_name(fup):
    n = sum(fup)
    if fup[1]==1 and fup[2]==0:   return "DRAW"
    if fup[1]==1 and fup[2]==1:   return "STOP"
    if n == 5:                     return "CLEAR"
    return "OTHER"


def stamp_shape(canvas, mode, pt1, pt2, color, thick):
    if mode == "RECT":
        cv2.rectangle(canvas, pt1, pt2, color, thick)
    elif mode == "CIRCLE":
        cx=(pt1[0]+pt2[0])//2; cy=(pt1[1]+pt2[1])//2
        r=int(((pt2[0]-pt1[0])**2+(pt2[1]-pt1[1])**2)**0.5//2)
        cv2.circle(canvas,(cx,cy),max(r,1),color,thick)
    elif mode == "LINE":
        cv2.line(canvas, pt1, pt2, color, thick)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=1,
        min_hand_detection_confidence=0.70,
        min_hand_presence_confidence=0.70,
        min_tracking_confidence=0.60,
    )
    landmarker = HandLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ret, frame = cap.read()
    if not ret:
        print("Cannot open camera"); return

    h, w     = frame.shape[:2]
    canvas   = np.zeros((h, w, 3), dtype=np.uint8)
    overlay  = None
    history  = History()
    pfilter  = PointFilter()
    g_stab   = GestureStabilizer(required=3)

    colors      = list(DEFAULT_COLORS) + [CUSTOM_SLOT]
    prev_pt     = None
    shape_pt1   = None
    mode        = "DRAW"
    sel_color   = 0
    brush_sz    = 15
    opacity     = 1.0
    is_eraser   = False

    text_mode   = False
    text_buf    = ""
    text_pos    = (100, 300)

    # Color picker state
    show_picker   = False
    picker_img    = make_color_picker(300)
    picker_x      = w//2 - 150
    picker_y      = h//2 - 150
    picker_dragging = False

    frame_count = 0
    last_result = None

    print("Air Canvas v3 ready!")
    print("1 finger=draw | 2 fingers=stop | 5 fingers=clear")
    print("'+CLR' button in toolbar = open color picker to set custom color")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # Detection every frame for better accuracy (skip every 3rd only if CPU struggles)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        last_result = landmarker.detect(mp_image)
        result = last_result

        # ── Blend canvas ───────────────────────────────────────────────────────
        if opacity >= 0.99:
            mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            frame[mask > 0] = canvas[mask > 0]
        else:
            alpha_c = canvas.astype(float) * opacity
            alpha_f = frame.astype(float) * (1.0 - opacity*(canvas>0).astype(float))
            blended = np.clip(alpha_c+alpha_f, 0, 255).astype(np.uint8)
            mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            frame[mask > 0] = blended[mask > 0]

        # Shape preview
        if overlay is not None:
            pm = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            _, pm = cv2.threshold(pm, 10, 255, cv2.THRESH_BINARY)
            frame[pm > 0] = overlay[pm > 0]

        # Text preview
        if text_mode and text_buf:
            cv2.putText(frame, text_buf+"|", text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors[sel_color][1], 2)

        # ── Sliders ────────────────────────────────────────────────────────────
        r_size, r_opac = draw_sliders(frame, h, brush_sz, opacity)
        sx1,sy1,sw1,sh1,smn,smx = r_size
        sx2,sy2,sw2,sh2,omn,omx = r_opac

        cur_color = ERASER_COLOR if is_eraser else colors[sel_color][1]

        # ── Color picker overlay ───────────────────────────────────────────────
        if show_picker:
            ph, pw = picker_img.shape[:2]
            # Draw picker window
            px1,py1 = picker_x, picker_y
            px2,py2 = picker_x+pw+20, picker_y+ph+60

            # Bounds check
            px1 = max(0, min(px1, w-pw-20))
            py1 = max(TOOLBAR_H, min(py1, h-ph-60))
            picker_x, picker_y = px1, py1

            sub = frame[py1:py1+ph+60, px1:px1+pw+20].copy()
            cv2.rectangle(frame,(px1,py1),(px1+pw+20,py1+ph+60),(30,30,40),-1)
            cv2.rectangle(frame,(px1,py1),(px1+pw+20,py1+ph+60),(80,80,100),2)
            frame[py1+10:py1+10+ph, px1+10:px1+10+pw] = picker_img

            # Title
            cv2.putText(frame,"CUSTOM COLOR PICKER",
                        (px1+10, py1+ph+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,210), 1)

            # Preview box
            cv2.rectangle(frame,(px1+10,py1+ph+36),(px1+pw//2,py1+ph+52),
                          colors[-1][1],-1)
            cv2.rectangle(frame,(px1+10,py1+ph+36),(px1+pw//2,py1+ph+52),(150,150,160),1)

            # Close button
            cv2.rectangle(frame,(px1+pw-10,py1+ph+36),(px1+pw+10,py1+ph+52),(60,40,40),-1)
            cv2.putText(frame,"X",(px1+pw-5,py1+ph+50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,80,80),1)

        # ── Hand processing ────────────────────────────────────────────────────
        hand_detected = False
        if result and result.hand_landmarks:
            for hand_lm in result.hand_landmarks:
                hand_detected = True
                lm_norm = [(lm.x, lm.y) for lm in hand_lm]
                lm_px   = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]

                # Skeleton
                for c in HAND_CONNECTIONS:
                    cv2.line(frame, lm_px[c[0]], lm_px[c[1]], (0,160,55), 1)
                for pt in lm_px:
                    cv2.circle(frame, pt, 3, (0,210,75), -1)

                fup = fingers_up(lm_norm)
                n   = sum(fup)
                raw_x, raw_y = lm_px[8]
                ix, iy = pfilter.update(raw_x, raw_y)

                g = g_stab.update(gesture_name(fup))

                # Cursor
                cv2.circle(frame,(ix,iy),brush_sz//2+4,cur_color,-1)
                cv2.circle(frame,(ix,iy),brush_sz//2+4,(200,200,200),1)

                # ── Picker interaction ────────────────────────────────────────
                if show_picker:
                    ph, pw2 = picker_img.shape[:2]
                    rel_x = ix - (picker_x+10)
                    rel_y = iy - (picker_y+10)
                    # Close btn
                    if (picker_x+pw2-10 < ix < picker_x+pw2+10 and
                        picker_y+ph+36 < iy < picker_y+ph+52 and fup[1]==1):
                        show_picker = False
                    # Sample wheel
                    elif 0 <= rel_x < pw2 and 0 <= rel_y < ph and fup[1]==1:
                        sampled = sample_wheel(picker_img, rel_x, rel_y)
                        colors[-1] = ("Custom", sampled)
                        sel_color  = len(colors)-1
                        is_eraser  = False
                    continue

                # ── Toolbar ───────────────────────────────────────────────────
                if iy < TOOLBAR_H and fup[1]==1:
                    _, icon_rects = make_toolbar(w,colors,sel_color,is_eraser,mode,brush_sz,opacity)
                    for rx1,_,rx2,_,tag in icon_rects:
                        if rx1 < ix < rx2:
                            kind,val = tag
                            if kind=="color":
                                sel_color=val; is_eraser=False
                            elif kind=="eraser":
                                is_eraser = not is_eraser
                            elif kind=="mode":
                                mode=val; is_eraser=False
                            elif kind=="picker":
                                show_picker=True
                    pfilter.reset(); prev_pt=None; shape_pt1=None
                    continue

                # ── Sliders ───────────────────────────────────────────────────
                if sx1<ix<sx1+sw1 and sy1<iy<sy1+sh1 and fup[1]==1:
                    ratio = 1.0-(iy-sy1)/sh1
                    brush_sz = int(smn+max(0,min(1,ratio))*(smx-smn))
                    prev_pt=None; continue
                if sx2<ix<sx2+sw2 and sy2<iy<sy2+sh2 and fup[1]==1:
                    ratio = 1.0-(iy-sy2)/sh2
                    opacity = round(omn/100+max(0,min(1,ratio))*(omx-omn)/100,2)
                    prev_pt=None; continue

                # ── Gestures ──────────────────────────────────────────────────
                if n==5:
                    history.push(canvas); canvas[:]=0
                    pfilter.reset(); prev_pt=None; continue

                if g == "DRAW" and iy > TOOLBAR_H:
                    if mode=="DRAW":
                        if prev_pt:
                            thick = brush_sz if not is_eraser else brush_sz*3
                            cv2.line(canvas, prev_pt,(ix,iy),cur_color,thick)
                        else:
                            history.push(canvas)
                        prev_pt=(ix,iy)
                    elif mode in ("RECT","CIRCLE","LINE"):
                        if shape_pt1 is None:
                            history.push(canvas); shape_pt1=(ix,iy)
                        else:
                            overlay=np.zeros_like(canvas)
                            stamp_shape(overlay,mode,shape_pt1,(ix,iy),cur_color,brush_sz)
                        prev_pt=None
                    elif mode=="TEXT" and text_mode:
                        text_pos=(ix,iy)

                elif g=="STOP" or (g=="OTHER" and n>=2):
                    if mode in ("RECT","CIRCLE","LINE") and shape_pt1:
                        stamp_shape(canvas,mode,shape_pt1,(ix,iy),cur_color,brush_sz)
                        shape_pt1=None; overlay=None
                    pfilter.reset(); prev_pt=None

        if not hand_detected:
            pfilter.reset(); prev_pt=None

        # ── Toolbar render ─────────────────────────────────────────────────────
        toolbar, _ = make_toolbar(w,colors,sel_color,is_eraser,mode,brush_sz,opacity)
        frame[:TOOLBAR_H] = cv2.resize(toolbar,(w,TOOLBAR_H))

        # ── HUD ────────────────────────────────────────────────────────────────
        tool = "ERASE" if is_eraser else f"{mode} [{colors[sel_color][0]}]"
        hud  = f"{tool}  sz:{brush_sz}  op:{int(opacity*100)}%  Z=undo Y=redo C=clear T=text Q=quit"
        cv2.putText(frame,hud,(15,TOOLBAR_H+26),
                    cv2.FONT_HERSHEY_SIMPLEX,0.52,
                    (75,215,75) if not is_eraser else (75,75,215),1)
        if text_mode:
            cv2.putText(frame,"TEXT MODE: type → Enter to stamp | Esc cancel",
                        (15,TOOLBAR_H+48),cv2.FONT_HERSHEY_SIMPLEX,0.48,(20,195,225),1)

        cv2.imshow("Air Canvas v3", frame)

        # ── Keys ───────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if text_mode:
            if key==13:
                cv2.putText(canvas,text_buf,text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX,1.2,cur_color,2)
                text_buf=""; text_mode=False
            elif key==27: text_buf=""; text_mode=False
            elif key==8:  text_buf=text_buf[:-1]
            elif 32<=key<=126: text_buf+=chr(key)
            continue

        if key in (ord('q'),27): break
        elif key in (ord('z'),ord('Z')): canvas=history.undo(canvas)
        elif key in (ord('y'),ord('Y')): canvas=history.redo(canvas)
        elif key in (ord('c'),ord('C')): history.push(canvas); canvas[:]=0
        elif key in (ord('e'),ord('E')): is_eraser=not is_eraser
        elif key in (ord('m'),ord('M')):
            mode=MODES[(MODES.index(mode)+1)%len(MODES)]
        elif key in (ord('t'),ord('T')):
            text_mode=True; mode="TEXT"
        elif key in (ord('p'),ord('P')):
            show_picker = not show_picker

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()