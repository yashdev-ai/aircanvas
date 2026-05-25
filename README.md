# ✋ Air Canvas – Hand Gesture Drawing

Draw in the air using only your webcam and hand gestures!

## Setup

```bash
pip install opencv-python mediapipe numpy
python air_canvas.py
```

## Controls

| Gesture / Key | Action |
|---|---|
| ☝️ 1 finger (index only) | **Draw** on canvas |
| ✌️ 2 fingers (index + middle) | **Lift pen** / move without drawing |
| 🖐️ Full palm (5 fingers) | **Clear** entire canvas |
| Click toolbar brush | **Change color** |
| Click ERASE icon | Switch to **eraser** |
| Drag right-side slider | Adjust **brush/eraser size** |
| `M` | Toggle DRAW / SHAPE mode |
| `C` | Clear canvas |
| `E` | Toggle eraser |
| `Q` or `ESC` | Quit |

## How It Works

- **MediaPipe Hands** tracks 21 hand landmarks in real time
- Index fingertip (landmark #8) is the drawing cursor
- Finger state is detected by comparing tip vs. PIP joint Y-coordinates
- Drawing is layered on a black canvas and blended onto the webcam feed
- Toolbar at top and size slider on the right are interacted with by moving your finger there

## Requirements

- Python 3.8+
- Webcam
- Good lighting for best hand detection
