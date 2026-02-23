# Depth Anything 3 - TouchDesigner Integration

Complete TouchDesigner setup for real-time depth estimation streaming.

## Features

- **Parameter-driven configuration** - All settings exposed as TD parameters
- **Auto-reconnect** - Automatic WebSocket reconnection with configurable URLs
- **Backend management** - Start/stop/restart the DA3 streaming server from TD
- **Frame skip** - Process every Nth frame for performance tuning
- **Colorized depth display** - Multiple colormaps with brightness/contrast controls
- **Model switching** - Change models on the fly with auto-restart
- **Device selection** - MPS, CUDA, or CPU backend
- **Configurable streaming** - Window size, overlap, FPS, quality, resolution

## Quick Start

### 1. Start the Backend (Option A: From Terminal)

```bash
cd /Users/flo/work/code/Depth-Anything-3
source .venv/bin/activate
da3 stream --model-dir depth-anything/DA3-BASE --device mps --port 8080
```

### 2. TouchDesigner Setup

#### Create the Network

1. **Add WebSocket DAT** (name it `websocket1`)
   - Address: `ws://localhost:8080/stream?window_size=8&overlap=2`
   - Auto Reconnect: `On`

2. **Add Text DAT** (name it `da3_stream_control`)
   - Copy contents of [da3_stream_control.py](da3_stream_control.py)
   - This creates the parameter interface

3. **Link WebSocket to Control Script**
   - Set WebSocket DAT parameter **Callbacks DAT** to: `da3_stream_control`

4. **Add chopexecDAT** (name it `frame_sender`)
   - Copy contents of [da3_send_frames.py](da3_send_frames.py)
   - Connect to a Timer CHOP (pulse mode, 30 FPS)

5. **Add Script TOP** (name it `depth_display`)
   - Copy contents of [da3_depth_display.py](da3_depth_display.py)
   - This will display the depth maps

6. **Add your video source** (name it `moviefilein1`)
   - Movie File In TOP, Video Device In TOP, or any TOP
   - Edit `da3_send_frames.py` line 35 if using different name

## Component Structure

```
┌─────────────────────────────────────────────────────────┐
│ TouchDesigner Network                                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Timer CHOP] ──> [chopexecDAT: frame_sender]          │
│       30fps              │                              │
│                          │                              │
│  [Movie File In TOP] <───┘                              │
│     moviefilein1                                        │
│                          │                              │
│                          ▼                              │
│              [WebSocket DAT: websocket1] <──────┐       │
│                          │                      │       │
│                          │                      │       │
│                          ▼                      │       │
│              [Text DAT: da3_stream_control] ────┘       │
│                          │                              │
│                          ├──> stores 'depth_data'       │
│                          │                              │
│                          ▼                              │
│              [Script TOP: depth_display]                │
│                          │                              │
│                          ▼                              │
│                   [Your compositing]                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Parameters (da3_stream_control)

### Connection

- **Host**: Server hostname (default: `localhost`)
- **Port**: Server port (default: `8080`)

### Model Configuration

- **Model**: Choose from DA3-SMALL, DA3-BASE, DA3-LARGE, DA3-GIANT, DA3NESTED-GIANT-LARGE
- **Device**: MPS (Apple Silicon), CUDA (NVIDIA), CPU

### Stream Configuration

- **Window Size**: Number of frames per batch (1-32, default: 8)
  - `1` = Real-time, lowest latency, most jitter
  - `8` = Balanced
  - `16-32` = Smoothest, higher latency
- **Overlap**: Overlapping frames between batches (0-16, default: 2)
- **Process Resolution**: Model processing resolution (252-1008, default: 504)
  - Must be divisible by 14 (ViT patch size)
  - Higher = better quality, slower
- **Max FPS**: Server-side FPS limit (1-60, default: 15)
- **JPEG Quality**: Compression quality (50-95, default: 75)

### Input Settings

- **Frame Skip (N)**: Process every Nth frame (1-10, default: 1)
  - `1` = Every frame
  - `2` = Every other frame (half speed)
  - `5` = Every 5th frame (20% speed)
- **Input Width/Height**: Expected input resolution

### Backend Management

- **Backend Path**: Path to DA3 repository
- **Model Directory**: HuggingFace model identifier
- **Backend Port**: Port for backend server
- **Auto Restart Backend**: Restart when model/device changes

### Controls

- **Reconnect**: Reconnect WebSocket with current settings
- **Restart Backend**: Kill and restart the streaming server
- **Stop Backend**: Stop the backend server

### Status (Read-only)

- **Status**: WebSocket connection status and FPS
- **Backend Status**: Backend server health and model info

## Parameters (depth_display)

### Visualization

- **Colorize Depth**: Enable color mapping (default: `On`)
- **Color Map**: Choose colormap
  - Viridis (Blue-Green-Yellow)
  - Plasma (Purple-Pink-Yellow)
  - Inferno (Black-Red-Yellow)
  - Magma (Black-Purple-White)
  - Turbo (Rainbow)
  - Grayscale
- **Invert Depth**: Flip near/far (default: `Off`)
- **Brightness**: Adjust brightness (0-3, default: 1.0)
- **Contrast**: Adjust contrast (0-3, default: 1.0)

## Performance Tuning

### For Real-time Interactive (Low Latency)

```
Model: DA3-SMALL
Window Size: 1
Overlap: 0
Process Resolution: 252 or 378
Max FPS: 30
Frame Skip: 1
Quality: 70
```

Expected: 15-25 FPS on MPS

### For High Quality (Smooth)

```
Model: DA3-BASE or DA3-LARGE
Window Size: 8
Overlap: 2
Process Resolution: 504
Max FPS: 15
Frame Skip: 1
Quality: 85
```

Expected: 5-15 FPS on MPS

### For Performance Testing

```
Model: DA3-SMALL
Window Size: 4
Overlap: 1
Process Resolution: 378
Max FPS: 20
Frame Skip: 2  (process every other frame)
Quality: 75
```

Expected: 10-20 FPS on MPS

## Workflow Examples

### Example 1: Live Webcam Depth

1. Use **Video Device In TOP** as source
2. Set Frame Skip to `1` (every frame)
3. Window Size `1` for real-time
4. Model: `DA3-SMALL`

### Example 2: Movie File Processing

1. Use **Movie File In TOP** as source
2. Set Frame Skip to `2` or `5` for faster processing
3. Window Size `8` for smooth depth
4. Model: `DA3-BASE`

### Example 3: Multi-view Reconstruction

1. Use **Movie File In TOP** with 10-20 frames
2. Set Frame Skip to `1`
3. Window Size `12` (entire sequence)
4. Model: `DA3NESTED-GIANT-LARGE` for metric depth

## Troubleshooting

### "WebSocket not connected"

- Check that backend server is running
- Verify Host/Port settings match backend
- Try clicking **Reconnect** button

### "Backend unreachable"

- Click **Restart Backend** button
- Or manually start from terminal
- Check Backend Path is correct

### Low FPS

- Reduce Process Resolution (e.g., 378 or 252)
- Increase Frame Skip (e.g., 2 or 5)
- Use smaller model (DA3-SMALL)
- Reduce Window Size to 1 or 4
- Lower JPEG Quality (e.g., 70)

### Jittery Depth

- Increase Window Size (e.g., 8 or 16)
- Increase Overlap (e.g., 3 or 4)
- Reduce Frame Skip to 1
- Use larger model (DA3-BASE or DA3-LARGE)

### Backend won't start from TD

- Check Backend Path is correct
- Verify venv exists at `.venv/bin/activate`
- Check Model Directory is valid
- Manually start from terminal to see errors

## Advanced: Manual Backend Control

If you prefer to manage the backend manually:

1. Set **Auto Restart Backend** to `Off`
2. Start backend from terminal:
   ```bash
   source .venv/bin/activate
   da3 stream --model-dir depth-anything/DA3-BASE --device mps --port 8080 --process-res 504
   ```
3. Use **Reconnect** button to update WebSocket URL parameters

## Files

- [da3_stream_control.py](da3_stream_control.py) - Main control script with parameters
- [da3_send_frames.py](da3_send_frames.py) - Frame sender (chopexecDAT)
- [da3_depth_display.py](da3_depth_display.py) - Depth visualization (Script TOP)
- [README.md](README.md) - This file

## Notes

- Backend process is stored in `parent().store('da3_backend_process')`
- Depth data is stored in `parent().store('depth_data')` as base64 JPEG
- Frame counter is stored in `parent().store('frame_counter')`
- Health checks run every 2 seconds (60 frames @ 30fps)

## Integration with Other Software

This same WebSocket protocol works with:
- **Processing**: Use websocket library
- **Max/MSP**: Use `jweb` or `node.script`
- **vvvv**: Use WebSocket nodes
- **Unity**: Use WebSocket client
- **OpenFrameworks**: Use ofxWebSocket addon

See [CLAUDE.md](../CLAUDE.md) for protocol details.
