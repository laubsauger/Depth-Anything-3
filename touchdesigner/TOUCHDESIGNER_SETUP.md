# TouchDesigner Setup (Numpy Version - Working)

Complete setup for real-time depth estimation in TouchDesigner using raw numpy arrays.

## Files

- **da3_stream_control_numpy.py** - Control script with parameters (Text DAT)
- **da3_send_frames_numpy.py** - Frame sender (chopexecDAT)
- **da3_depth_display_numpy.py** - Depth visualization (Script TOP)

## Quick Setup

### 1. Start Backend

```bash
cd /Users/flo/work/code/Depth-Anything-3
source .venv/bin/activate
da3 stream --model-dir depth-anything/DA3-BASE --device mps --port 8080
```

### 2. TouchDesigner Network

Create these operators:

```
[Timer CHOP]
     |
     v
[chopexecDAT: frame_sender] ----reads----> [TOP: in_video]
     |
     v (sends frames)
[WebSocket DAT: websocket1] <----callbacks---- [Text DAT: da3_stream_control]
     |
     v (receives depth)
[Script TOP: depth_display]
```

### 3. Create Operators

**A. Text DAT** (name: `da3_stream_control`)
- Create Text DAT
- Copy contents of `da3_stream_control_numpy.py`
- This adds all parameters automatically

**B. WebSocket DAT** (name: `websocket1`)
- Create WebSocket DAT
- Set **Callbacks DAT** to: `da3_stream_control`
- Leave **Address** blank (set by control script)

**C. Script TOP** (name: `depth_display`)
- Create Script TOP
- Copy contents of `da3_depth_display_numpy.py`
- This will show depth maps

**D. chopexecDAT** (name: `frame_sender`)
- Create chopexecDAT
- Copy contents of `da3_send_frames_numpy.py`

**E. Timer CHOP** (name: `timer1`)
- Create Timer CHOP
- **Play**: On
- **Pulse**: On
- **Length**: 1.0
- **Cycle**: On
- **FPS**: 30

**F. Video Input** (name: `in_video`)
- Create your video source (Movie File In TOP, Video Device In, etc.)
- **MUST be named `in_video`**

### 4. Connect

1. Connect Timer CHOP output to chopexecDAT input
2. Set chopexecDAT parameter **CHOP** to: `timer1`

### 5. Configure & Connect

In `da3_stream_control` parameters:
- **Host**: `localhost`
- **Port**: `8080`
- **Model**: `DA3-BASE`
- **Device**: `mps`
- **Quality**: `100` (raw float32)
- **Window Size**: `8`
- **Overlap**: `2`

Click **Reconnect** button.

## Parameters

### da3_stream_control

**Connection:**
- Host: `localhost`
- Port: `8080`

**Model:**
- Model: DA3-SMALL / DA3-BASE / DA3-LARGE
- Device: mps / cuda / cpu

**Stream:**
- Window Size: 1-32 (default: 8)
- Overlap: 0-16 (default: 2)
- Process Resolution: 252-1008 (default: 504)
- Max FPS: 1-60 (default: 15)
- Quality: 50-100 (100 = raw float32, <100 = JPEG)

**Input:**
- Frame Skip: 1-10 (1=every frame, 2=every other)
- Send as Numpy: On/Off

**Controls:**
- **Reconnect**: Connect WebSocket
- **Restart Backend**: Restart server
- **Stop Backend**: Stop server

### depth_display

**Visualization:**
- Auto Normalize: On/Off
- Colorize Depth: On/Off
- Color Map: viridis / plasma / inferno / magma / turbo / gray
- Invert Depth: On/Off
- Brightness: 0-3
- Contrast: 0-3

## Key Differences from Old Version

1. ✅ Uses `copyNumpyArray()` not `copyNPixels()`
2. ✅ Parameters set via `p[0].default = value` not `default=value`
3. ✅ Uses RGBA float32 format `(H, W, 4)` in range `[0, 1]`
4. ✅ Video input must be named `in_video`
5. ✅ No PIL dependency - pure numpy

## Troubleshooting

**"No module named 'PIL'"**
- Make sure you're using `da3_depth_display_numpy.py` not the old version
- This version doesn't use PIL

**"object has no attribute 'copyNPixels'"**
- Update to latest `da3_depth_display_numpy.py`
- Uses `copyNumpyArray()` instead

**"Call contains unexpected keywords: default"**
- Update to latest version - parameters now set correctly

**No depth appearing**
- Check **Status** shows "Connected"
- Check backend is running: `curl http://localhost:8080/health`
- Check `parent().fetch('depth_array')` in textport

**"ERROR: No operator in_video"**
- Rename your video source TOP to `in_video`
- Or edit line 39 in `da3_send_frames_numpy.py`

## Performance

### Real-time (Low Latency)
```
Model: DA3-SMALL
Window Size: 1
Overlap: 0
Quality: 100
Frame Skip: 1
```
Expected: 15-25 FPS on M1/M2

### Balanced
```
Model: DA3-BASE
Window Size: 8
Overlap: 2
Quality: 100
Frame Skip: 1
```
Expected: 5-15 FPS on M1/M2

### High Quality
```
Model: DA3-LARGE
Window Size: 16
Overlap: 4
Quality: 100
Frame Skip: 2
```
Expected: 2-8 FPS on M1/M2

## Data Flow

```
in_video (TOP)
    |
    v numpyArray()
frame_sender (chopexecDAT)
    |
    v JSON metadata + binary numpy
WebSocket (sends to server)
    |
    v processes depth
WebSocket (receives from server)
    |
    v JSON + base64 numpy
da3_stream_control (onReceiveText)
    |
    v stores in parent()
    |
    v parent().store('depth_array', depth)
    |
depth_display (Script TOP)
    |
    v parent().fetch('depth_array')
    |
    v apply colormap
    |
    v copyNumpyArray(rgba_float32)
    |
Output (depth visualization)
```

## Notes

- Raw float32 mode (`quality=100`) gives full precision depth
- JPEG mode (`quality<100`) compresses to uint8 (smaller, faster, lossy)
- Window size affects temporal smoothness vs latency tradeoff
- Frame skip reduces processing load for testing
