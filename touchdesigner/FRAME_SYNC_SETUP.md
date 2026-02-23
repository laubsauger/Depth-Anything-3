# Frame Synchronization Setup

This document explains how to set up frame synchronization between input RGB and processed depth maps in TouchDesigner.

## Problem

Due to buffering, network latency, and processing time, there's a delay between sending an RGB frame and receiving its corresponding depth map. Without synchronization, you can't reliably match RGB frames to their depth outputs.

## Solution

We use a **sequence ID tagging system** with a **circular buffer**:

1. Each sent frame gets a unique sequence ID
2. RGB frames are stored in a circular buffer with their sequence IDs
3. When depth arrives, it includes the sequence ID of the RGB frame it was generated from
4. We retrieve the matching RGB frame from the buffer

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Frame Flow                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  in_video (TOP)                                                      │
│       │                                                               │
│       ├────────────────────────────┐                                │
│       │                            │                                │
│       ▼                            ▼                                │
│  send_frame()                 Store in circular buffer              │
│       │                       (seq_id → RGB frame)                  │
│       │                                                               │
│       ▼                                                               │
│  WebSocket Send                                                      │
│  (seq_id: 123)                                                       │
│       │                                                               │
│       ▼                                                               │
│  ┌──────────────┐                                                    │
│  │ DA3 Backend  │ Processing...                                     │
│  └──────────────┘                                                    │
│       │                                                               │
│       ▼                                                               │
│  WebSocket Receive                                                   │
│  (seq_id: 123, depth_map)                                           │
│       │                                                               │
│       ▼                                                               │
│  onReceiveText()                                                     │
│       │                                                               │
│       ├──────────────────────────────────────────┐                  │
│       │                                          │                  │
│       ▼                                          ▼                  │
│  Store depth_array                  Retrieve RGB for seq_id: 123   │
│       │                                          │                  │
│       ▼                                          ▼                  │
│  out_depth (Script TOP)            out_synched (Script TOP)         │
│       │                                          │                  │
│       ▼                                          ▼                  │
│  Depth visualization               Synced RGB frame                 │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Setup Instructions

### 1. Add Frame Sync Script

1. Create a Text DAT named **`da3_frame_sync`**
2. Copy contents from `touchdesigner/da3_frame_sync.py`

### 2. Update Extension Parameters

The extension automatically adds the `Syncbuffersize` parameter (default: 30 frames).

This parameter controls how many frames are kept in the circular buffer:
- **Larger buffer**: Can handle longer delays, uses more memory
- **Smaller buffer**: Lower latency, may miss frames if processing is slow

### 3. Create Output Script TOPs

Create three Script TOPs:

#### a) **out_input** (Latest Input)
- Create Script TOP named `out_input`
- Set **Script** parameter to `da3_out_input` (Text DAT)
- Shows: Most recently captured RGB frame (before sending)

#### b) **out_depth** (Depth Map)
- This is your existing `depth_display` Script TOP
- Shows: Processed depth map with colorization

#### c) **out_synched** (Synchronized RGB)
- Create Script TOP named `out_synched`
- Set **Script** parameter to `da3_out_synched` (Text DAT)
- Shows: RGB frame that matches the current depth frame

### 4. Required DAT Files

Ensure these Text DATs exist in your COMP:

- `da3_extension` → Extension script
- `da3_stream_control` → WebSocket callbacks
- `da3_send_frames_numpy` → Frame sender
- `da3_frame_sync` → **NEW** - Frame synchronization manager
- `da3_depth_display_numpy` → Depth visualization
- `da3_out_input` → **NEW** - Latest input output
- `da3_out_synched` → **NEW** - Synced RGB output

## How It Works

### Sending Frames (da3_send_frames_numpy.py)

```python
# Store frame in sync manager with unique sequence ID
sync_mgr = sync_dat.module.get_sync_manager(comp)
seq_id = sync_mgr.store_sent_frame(frame_array)

# Send metadata with sequence ID
metadata = {
    "format": "numpy",
    "shape": [h, w, c],
    "dtype": "uint8",
    "seq_id": seq_id  # ← Tagged with sequence ID
}
websocket_dat.sendText(json.dumps(metadata))
```

### Receiving Depth (da3_stream_control_numpy.py)

```python
# Extract sequence ID from response
seq_id = data.get('seq_id', None)

# Retrieve matching RGB frame
sync_mgr = sync_dat.module.get_sync_manager(comp)
synced_rgb = sync_mgr.get_frame_for_depth(seq_id)

# Store for Script TOP to access
comp.store('synced_rgb_frame', synced_rgb)

# Trigger output to cook
comp.op('out_synched').cook(force=True)
```

### Circular Buffer

The frame buffer is a **deque** with maximum size:
- Old frames automatically age out when buffer is full
- Default size: 30 frames (~1-2 seconds at 15-30 FPS)
- Configurable via `Syncbuffersize` parameter

## Server-Side Changes Required

**IMPORTANT**: The streaming server must be modified to:

1. **Accept `seq_id` in incoming metadata**
2. **Echo `seq_id` back in the response**

Example server modification:

```python
# In stream_server.py - process_frame endpoint
async def process_frame(websocket, message):
    # Parse incoming metadata
    metadata = json.loads(message)
    seq_id = metadata.get('seq_id', None)

    # ... process frame ...

    # Echo seq_id back in response
    response = {
        'status': 'success',
        'data': depth_b64,
        'shape': depth.shape,
        'format': 'raw_float32',
        'seq_id': seq_id  # ← Echo back the sequence ID
    }
    await websocket.send(json.dumps(response))
```

## Outputs

### out_input
- **What**: Latest captured RGB frame
- **Resolution**: Same as `in_video` (after resolution_preprocess)
- **Update**: Every time a frame is sent to server
- **Use case**: Preview what's being sent, debug input pipeline

### out_depth
- **What**: Processed depth map (colorized)
- **Resolution**: Server processing resolution
- **Update**: When depth result arrives from server
- **Use case**: Visualize depth estimation

### out_synched
- **What**: RGB frame matching current depth
- **Resolution**: Same as `in_video` (after resolution_preprocess)
- **Update**: When depth result arrives from server
- **Use case**: Composite RGB + depth, side-by-side comparison, mask generation

## Troubleshooting

### "No RGB frame found for seq=X (aged out?)"

**Cause**: Frame aged out of circular buffer before depth arrived.

**Solutions**:
1. Increase `Syncbuffersize` parameter (e.g., 60 frames)
2. Reduce server processing time (use smaller model like DA3-SMALL)
3. Reduce `window_size` parameter (e.g., set to 1 for real-time)

### out_synched shows old/stale frames

**Cause**: Server not echoing `seq_id` back, falling back to last known frame.

**Solution**: Verify server-side code echoes `seq_id` in responses.

### Memory usage is high

**Cause**: Large `Syncbuffersize` with high-resolution frames.

**Solutions**:
1. Reduce `Syncbuffersize` parameter
2. Use lower resolution input (via `resolution_preprocess` TOP)

## Example Network

```
[Video Source]
    │
    ▼
[resolution_preprocess] (Resolution TOP)
    │
    ▼
[in_video] (Null TOP)
    │
    ├──────────────────────┬───────────────────┐
    │                      │                   │
    ▼                      ▼                   ▼
[DepthAnything3 COMP]  [out_input]      [Composite TOP]
    │                                           │
    ├────────────────┬──────────────────────────┤
    │                │                          │
    ▼                ▼                          ▼
[out_depth]    [out_synched]           [Side-by-side view]
```

## Performance Considerations

### Buffer Size Guidelines

| FPS | Latency | Buffer Size |
|-----|---------|-------------|
| 30  | ~1s     | 30          |
| 15  | ~2s     | 30          |
| 30  | ~2s     | 60          |
| 15  | ~4s     | 60          |

**Rule of thumb**: `buffer_size = FPS × max_expected_latency_seconds`

### Memory Usage

Each stored frame uses: `width × height × 4 bytes × buffer_size`

Example:
- 256×144 @ 30 frames = ~4.4 MB
- 504×284 @ 30 frames = ~17.2 MB
- 1008×568 @ 30 frames = ~68.8 MB

## Advanced: Custom Sync Logic

You can extend `FrameSyncManager` for custom sync behavior:

```python
class CustomSyncManager(FrameSyncManager):
    def get_frame_for_depth(self, seq_id):
        # Custom logic: interpolate between frames
        # or use temporal smoothing
        return super().get_frame_for_depth(seq_id)
```

Then modify `get_sync_manager()` in `da3_frame_sync.py` to use your custom class.
