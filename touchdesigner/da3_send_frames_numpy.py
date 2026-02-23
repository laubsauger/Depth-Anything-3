"""
Frame Sender for DA3 Streaming (chopexecDAT) - Numpy Version

Sends frames as raw numpy arrays instead of JPEG for better performance.
Reads ALL parameters from parent COMP.

Usage:
1. Create a COMP container with parameters set up
2. Add this as chopexecDAT inside the COMP
3. Connect to a CHOP trigger (e.g., Timer CHOP)
"""


def onOffToOn(channel, sampleIndex, val, prev):
    """
    Called when channel value goes from 0 to non-zero.
    Use this with a Timer CHOP set to pulse mode.
    """
    send_frame()


def onValueChange(channel, sampleIndex, val, prev):
    """
    Alternative: send on every value change.
    Use this with a constant timer.
    """
    # Uncomment to send every frame instead of on pulse:
    # send_frame()
    pass


def send_frame():
    """
    Capture frame from TOP and send to WebSocket as raw numpy array.
    """
    # Get parent COMP
    comp = parent()

    # Get operators
    websocket_dat = comp.op('websocket1')
    source_top = comp.op('in_video')

    if not websocket_dat:
        debug("ERROR: No WebSocket DAT named 'websocket1' found!")
        return

    if not source_top:
        debug("ERROR: No video input TOP named 'in_video' found!")
        return

    # Get frame skip setting from parent
    frame_skip = comp.par.Frameskip.eval()

    # Check if we should skip this frame (only if frame_skip > 0)
    if frame_skip > 0:
        frame_counter = comp.fetch('frame_counter', 0)
        comp.store('frame_counter', frame_counter + 1)

        if frame_counter % (frame_skip + 1) != 0:
            return  # Skip this frame

    # Check if WebSocket is connected
    if not websocket_dat.par.active.eval():
        debug("WARNING: WebSocket not connected, skipping frame")
        return

    # Check if we should send as numpy or JPEG
    use_numpy = comp.par.Usenumpy.eval()

    try:
        if use_numpy:
            # Send as raw numpy array (faster, no compression artifacts)
            import numpy as np
            import json

            # Get numpy array from TOP
            try:
                frame_array = source_top.numpyArray(delayed=False)
                # TouchDesigner returns (height, width, channels) in float32 [0, 1]

                # Store frame for synchronization
                sync_dat = comp.op('da3_frame_sync')
                if sync_dat:
                    sync_mgr = sync_dat.module.get_sync_manager(comp)
                    seq_id = sync_mgr.store_sent_frame(frame_array)
                else:
                    seq_id = comp.fetch('frame_counter', 0)

                # Convert to uint8 [0, 255]
                frame_uint8 = (frame_array * 255).astype(np.uint8)

                h, w, c = frame_uint8.shape

                # Send JSON metadata first with sequence ID
                metadata = {
                    "format": "numpy",
                    "shape": [h, w, c],
                    "dtype": "uint8",
                    "seq_id": seq_id
                }
                websocket_dat.sendText(json.dumps(metadata))

                # Then send binary data
                websocket_dat.sendBinary(frame_uint8.tobytes())

                debug(f"Sent numpy frame: {h}x{w}x{c} [seq={seq_id}]")

            except AttributeError:
                # numpyArray() not available, fall back to JPEG
                debug("WARNING: numpyArray() not available, falling back to JPEG")
                send_frame_jpeg(source_top, websocket_dat, comp)

        else:
            # Send as JPEG (smaller, but lossy)
            send_frame_jpeg(source_top, websocket_dat, comp)

        # Update stats on parent
        comp.par.Framessent = comp.par.Framessent.eval() + 1

        # Trigger next frame load
        trigger_op = comp.op('trigger1')
        if trigger_op:
            trigger_op.par.triggerpulse.pulse()

    except Exception as e:
        debug(f"ERROR: Error sending frame: {e}")
        import traceback
        traceback.print_exc()


def send_frame_jpeg(source_top, websocket_dat, comp):
    """
    Send frame as JPEG (fallback method).
    """
    # Get JPEG quality from parent (or use default 85)
    try:
        quality = comp.par.Quality.eval()
        if quality == 100:
            quality = 85  # JPEG doesn't support 100 well
        quality_float = quality / 100.0  # Convert to 0.0-1.0
    except:
        quality_float = 0.85

    # Convert TOP to JPEG bytes
    jpeg_bytes = source_top.saveByteArray('.jpg', quality=quality_float)

    # Send via WebSocket as binary
    websocket_dat.sendBinary(jpeg_bytes)

    debug(f"Sent JPEG frame: {len(jpeg_bytes)} bytes")


def debug(msg):
    """Print debug message."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Frame Sender: {msg}")
