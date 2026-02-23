# Simple TouchDesigner Setup (No PIL Required)

This is the simplified setup that doesn't require PIL or numpy.

## Components Needed

1. **Text DAT** named `da3_stream_control`
2. **WebSocket DAT** named `websocket1`
3. **Movie File In TOP** named `depth_display`
4. **chopexecDAT** named `frame_sender`
5. **Movie File In TOP** for your video source (e.g., `moviefilein1`)
6. **Timer CHOP** to trigger frame sends

## Step-by-Step Setup

### 1. Create Control Script (Text DAT)

1. Create a **Text DAT**
2. Name it: `da3_stream_control`
3. Copy contents of [da3_stream_control.py](da3_stream_control.py)

This will create all the parameters automatically.

### 2. Create WebSocket Connection

1. Create a **WebSocket DAT**
2. Name it: `websocket1`
3. Set **Callbacks DAT** to: `da3_stream_control`
4. Leave **Address** blank (will be set by control script)

### 3. Create Depth Display (Movie File In TOP)

1. Create a **Movie File In TOP**
2. Name it: `depth_display`
3. Leave **File** parameter blank (will be updated by control script)
4. This will automatically display the depth maps

### 4. Create Frame Sender (chopexecDAT)

1. Create a **chopexecDAT**
2. Name it: `frame_sender`
3. Copy contents of [da3_send_frames.py](da3_send_frames.py)

### 5. Create Timer CHOP

1. Create a **Timer CHOP**
2. Set **Play** to: On
3. Set **Pulse** to: On
4. Set **Length** to: 1.0
5. Set **Cycle** to: On
6. Set **FPS** to: 30 (or desired frame rate)

### 6. Connect Timer to Frame Sender

1. In the **chopexecDAT** (`frame_sender`)
2. Set **CHOP** parameter to: `timer1` (or your Timer CHOP name)

### 7. Configure Your Video Source

1. Create or use existing **Movie File In TOP** or **Video Device In TOP**
2. Name it: `moviefilein1`
3. Or edit line 40 in `da3_send_frames.py` to match your TOP name

## Network Layout

```
[Timer CHOP: timer1]
       |
       v
[chopexecDAT: frame_sender] ----reads----> [Movie File In TOP: moviefilein1]
       |                                             (your video source)
       |
       v (sends frames)
[WebSocket DAT: websocket1] <----callbacks---- [Text DAT: da3_stream_control]
       |                                                    (parameters)
       v (receives depth)
[Movie File In TOP: depth_display]
       (shows depth maps)
```

## Usage

### Starting the Backend

**Option 1: From TouchDesigner**
1. Set parameters in `da3_stream_control`:
   - Host: `localhost`
   - Port: `8080`
   - Model: `DA3-BASE`
   - Device: `mps` (or `cuda`)
2. Click **Restart Backend** button

**Option 2: From Terminal**
```bash
cd /Users/flo/work/code/Depth-Anything-3
source .venv/bin/activate
da3 stream --model-dir depth-anything/DA3-BASE --device mps --port 8080
```

### Connecting

1. Click **Reconnect** button in `da3_stream_control` parameters
2. Watch **Status** parameter - should show "Connected"
3. Depth maps will appear in `depth_display` TOP

## Parameters

All parameters are in `da3_stream_control`:

### Connection
- **Host**: `localhost`
- **Port**: `8080`

### Model
- **Model**: Choose DA3-SMALL, DA3-BASE, etc.
- **Device**: Choose mps, cuda, or cpu

### Stream Configuration
- **Window Size**: 1-32 (default: 8)
- **Overlap**: 0-16 (default: 2)
- **Process Resolution**: 252-1008 (default: 504)
- **Max FPS**: 1-60 (default: 15)
- **JPEG Quality**: 50-95 (default: 75)

### Input
- **Frame Skip**: 1-10 (default: 1)
  - 1 = every frame
  - 2 = every other frame
  - 5 = every 5th frame

### Controls
- **Reconnect**: Reconnect WebSocket with new settings
- **Restart Backend**: Restart the streaming server
- **Stop Backend**: Stop the server

## Post-Processing the Depth Display

Since `depth_display` is just a Movie File In TOP, you can:

1. **Add Level TOP** after it to adjust brightness/contrast
2. **Add Lookup TOP** to apply color mapping
3. **Add Composite TOP** to blend with original video
4. **Add Blur TOP** for smoothing

Example:
```
[depth_display] --> [Level TOP] --> [Lookup TOP] --> [Composite TOP]
                                            |              ^
                                            v              |
                                    [Ramp TOP (colormap)]  |
                                                           |
                    [moviefilein1] -----------------------+
```

## Performance Tips

### Real-time Low Latency
- Window Size: 1
- Overlap: 0
- Process Resolution: 252 or 378
- Frame Skip: 1
- Model: DA3-SMALL

### High Quality Smooth
- Window Size: 8
- Overlap: 2
- Process Resolution: 504
- Frame Skip: 1
- Model: DA3-BASE

### Testing/Fast
- Window Size: 4
- Frame Skip: 2 or 5
- Process Resolution: 378
- Model: DA3-SMALL

## Troubleshooting

### No depth appearing
- Check **Status** shows "Connected"
- Check temp file exists: `/tmp/da3_depth_latest.jpg`
- Check `depth_display` TOP has a valid file path

### Low FPS
- Increase **Frame Skip**
- Reduce **Process Resolution**
- Use smaller model (DA3-SMALL)
- Reduce **Window Size** to 1

### Backend won't start
- Check **Backend Path** is correct
- Verify venv exists
- Start manually from terminal to see errors

## Files

- [da3_stream_control.py](da3_stream_control.py) - Main control (no PIL needed)
- [da3_send_frames.py](da3_send_frames.py) - Frame sender
- ~~[da3_depth_display.py](da3_depth_display.py)~~ - NOT NEEDED (use Movie File In TOP instead)
