# TouchDesigner Setup Guide - Depth Anything 3 Streaming

Complete setup instructions for real-time depth estimation in TouchDesigner using 32-bit float numpy arrays.

## Architecture Overview

The system uses a parent COMP with an extension that manages all parameters and control. Inside the COMP:
- **DepthAnything3Ext** (extension) - Sets up parameters and provides control methods
- **da3_stream_control** (Text DAT) - WebSocket callbacks and backend management
- **da3_send_frames** (chopexecDAT) - Sends video frames to server
- **depth_display** (Script TOP) - Displays depth maps with colorization
- **websocket1** (WebSocket DAT) - WebSocket connection
- **in_video** (TOP) - Video input source
- **trigger1** (null/select) - Frame cache trigger

All parameters live on the parent COMP for easy access. All scripts read from parent parameters.

## Step-by-Step Setup

### 1. Create Parent COMP

1. Create a **Container COMP** in your network
2. Name it `DepthAnything3` (or whatever you prefer)
3. Open the COMP by double-clicking it

### 2. Add Extension

1. Inside the COMP, create a **Text DAT**
2. Name it `DepthAnything3Ext`
3. Paste the contents of `da3_extension.py` into it
4. Close the Text DAT

### 3. Configure Extension Object

1. Select the parent COMP (go back up one level)
2. In the parameters, find the **Extensions** section
3. Set **Extension Object** to:
   ```python
   op('./DepthAnything3Ext').module.DepthAnything3Ext(me)
   ```
4. You should now see 7 custom parameter pages appear on the COMP:
   - Connection
   - Model
   - Stream
   - Input
   - Visualization
   - Backend
   - Control
   - Status

### 4. Add Control Script

1. Go back inside the COMP
2. Create a **Text DAT**
3. Name it `da3_stream_control`
4. Paste the contents of `da3_stream_control_numpy.py` into it

### 5. Add WebSocket

1. Create a **WebSocket DAT**
2. Name it `websocket1`
3. Set **Callbacks DAT** parameter to: `da3_stream_control`
4. Leave **Active** unchecked for now (we'll connect later)

### 6. Add Video Input

1. Create your video source TOP (Movie File In, Video Device In, etc.)
2. Name it `in_video`
3. Configure it to play your video or capture from camera

### 7. Add Frame Trigger

1. Create a **Null TOP** or **Select TOP**
2. Name it `trigger1`
3. This will be pulsed to load the next frame from cache

### 8. Add Frame Sender

1. Create a **Timer CHOP**
2. Set it to pulse at your desired frame rate (e.g., 10 FPS)
3. Create a **chopexecDAT**
4. Name it `da3_send_frames`
5. Paste the contents of `da3_send_frames_numpy.py` into it
6. Set the chopexecDAT's **CHOP** parameter to point to the Timer CHOP

### 9. Add Depth Display

1. Create a **Script TOP**
2. Name it `depth_display`
3. Paste the contents of `da3_depth_display_numpy.py` into it
4. The Script TOP will automatically show depth maps when they arrive

### 10. Parameter Callbacks (Optional but Recommended)

To make the Control page pulse buttons work:

1. Select the parent COMP
2. Add a **Parameter Execute DAT**
3. Set **Parameters** to: `Reconnect Restartbackend Stopbackend`
4. Add this code:

```python
def onValueChange(par, prev):
    """Called when parameter changes."""
    comp = par.owner

    if par.name == 'Reconnect':
        comp.Reconnect()
    elif par.name == 'Restartbackend':
        comp.RestartBackend()
    elif par.name == 'Stopbackend':
        comp.StopBackend()
```

## Network Layout

```
DepthAnything3 (COMP)
├── DepthAnything3Ext (Text DAT) - Extension class
├── da3_stream_control (Text DAT) - WebSocket callbacks
├── da3_send_frames (chopexecDAT) - Frame sender
├── depth_display (Script TOP) - Depth visualization
├── websocket1 (WebSocket DAT) - WebSocket connection
├── in_video (Movie File In / Video Device In) - Video source
├── trigger1 (Null TOP) - Frame cache trigger
└── timer1 (Timer CHOP) - Frame send trigger
```

## Starting the Backend

### Option 1: Manual Start (Recommended for first time)

Open terminal and run:

```bash
cd /Users/flo/work/code/Depth-Anything-3
source .venv/bin/activate
da3 stream --model-dir depth-anything/DA3-BASE --device mps --port 8080
```

### Option 2: Auto-Start from TouchDesigner

1. Configure **Backend** parameters on the COMP:
   - **Backend Path**: `/Users/flo/work/code/Depth-Anything-3`
   - **Model Directory**: `depth-anything/DA3-BASE`
   - **Backend Port**: `8080`
   - **Auto Restart Backend**: On
2. Click **Restart Backend** button in **Control** page
3. Wait ~10 seconds for model to load
4. Check **Backend Status** in **Status** page

## Connecting

1. Configure **Connection** parameters:
   - **Host**: `localhost`
   - **Port**: `8080`
2. Click **Reconnect** button in **Control** page
3. **Connection Status** should change to "Connected"

## Configuring Stream Settings

Adjust **Stream** parameters for performance vs quality:

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| **Window Size** | Frames per processing window | 8 (MPS), 24 (CUDA) |
| **Overlap** | Overlapping frames for smoothness | 2-4 |
| **Process Resolution** | Depth map resolution | 504 (balanced) |
| **Max FPS** | Server-side FPS limit | 15 |
| **Quality** | 100 = raw float32, <100 = JPEG | 100 |

## Configuring Visualization

Adjust **Visualization** parameters:

- **Auto Normalize**: Normalize each frame independently
- **Colorize Depth**: Apply color mapping
- **Color Map**: viridis, plasma, inferno, magma, turbo, gray
- **Invert Depth**: Flip near/far
- **Brightness**: Multiply depth values (0.0-3.0)
- **Contrast**: Adjust contrast around midpoint (0.0-3.0)

## Troubleshooting

### "No module named 'numpy'" in TouchDesigner

TouchDesigner includes numpy - this shouldn't happen. If it does, check TouchDesigner's Python version.

### "Cannot use an extension during its initialization"

This happens if you try to access extension parameters too early. Use this pattern in expressions:
```python
parent().par.Host if parent().extensionsReady else 'localhost'
```

### WebSocket not connecting

1. Check backend is running: `curl http://localhost:8080/health`
2. Check Host and Port parameters
3. Look for errors in TouchDesigner Textport (Alt+T)

### Frames not sending

1. Check Timer CHOP is active and pulsing
2. Check `in_video` operator exists and has valid input
3. Check WebSocket is connected
4. Look for errors in Textport

### Depth display is black

1. Check **Status** page shows frames received > 0
2. Check `depth_display` Script TOP for errors (hover over it)
3. Try toggling **Auto Normalize** on
4. Check Textport for Python errors

### Backend crashes or out of memory

1. Reduce **Window Size** (try 4 or 1 for real-time)
2. Reduce **Process Resolution** (try 252)
3. Use smaller model (DA3-SMALL instead of DA3-BASE)
4. Reduce video input resolution

## Performance Tips

### Real-time Mode (Lowest Latency)
```
Window Size: 1
Overlap: 0
Max FPS: 30
Process Resolution: 252
Model: DA3-SMALL
```

### Balanced Mode
```
Window Size: 8
Overlap: 2
Max FPS: 15
Process Resolution: 504
Model: DA3-BASE
```

### High Quality Mode (Slow)
```
Window Size: 12
Overlap: 4
Max FPS: 10
Process Resolution: 1008
Model: DA3-LARGE
```

## Advanced: Accessing Parameters in Other Operators

All parameters are on the parent COMP, so you can reference them from anywhere:

In parameter expressions:
```python
parent().par.Brightness
parent().par.Connectionstatus
```

In scripts:
```python
comp = op('/project1/DepthAnything3')
fps = comp.par.Currentfps.eval()
brightness = comp.par.Brightness.eval()
```

## Advanced: Extension Methods

The extension provides these methods:

```python
comp = op('/project1/DepthAnything3')
comp.Reconnect()        # Reconnect WebSocket
comp.RestartBackend()   # Restart backend server
comp.StopBackend()      # Stop backend server
```

## File Reference

- `da3_extension.py` - Main extension class
- `da3_stream_control_numpy.py` - WebSocket control
- `da3_send_frames_numpy.py` - Frame sender
- `da3_depth_display_numpy.py` - Depth visualization
- `SETUP_GUIDE.md` - This file
