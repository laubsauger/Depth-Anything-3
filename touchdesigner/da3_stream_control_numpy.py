"""
Depth Anything 3 Streaming Control for TouchDesigner (Raw Numpy Version)

Uses raw numpy arrays instead of JPEG encoding for better performance.
Reads ALL parameters from parent COMP.

Usage:
1. Create a COMP container
2. Add da3_parent_params.py as Text DAT and set up parameters
3. Add this script as Text DAT named 'da3_stream_control'
4. Create a WebSocket DAT and set its Callbacks DAT to this script
"""


def build_websocket_url(comp):
    """
    Build WebSocket URL from parent parameters.

    Returns:
        str: WebSocket URL with query parameters
    """
    host = comp.par.Host.eval()
    port = comp.par.Port.eval()

    # Build query parameters
    params = []
    params.append(f"window_size={comp.par.Windowsize.eval()}")
    params.append(f"overlap={comp.par.Overlap.eval()}")
    params.append(f"max_fps={comp.par.Maxfps.eval()}")
    params.append(f"quality={comp.par.Quality.eval()}")

    query_string = '&'.join(params)
    url = f"ws://{host}:{port}/stream?{query_string}"

    debug(f"WebSocket URL: {url}")
    return url


def reconnect(comp):
    """
    Reconnect the WebSocket with current parameters.
    """
    ws = comp.op('websocket1')
    if not ws:
        debug("ERROR: No WebSocket DAT named 'websocket1' found!")
        comp.par.Connectionstatus = "Error: No WebSocket"
        return

    # Disconnect if connected
    if ws.par.active.eval():
        debug("Disconnecting existing WebSocket...")
        ws.par.active = False
        # Wait a frame before reconnecting
        run("parent().do_connect()", delayFrames=30)
    else:
        do_connect(comp)


def do_connect(comp):
    """Actually perform the connection (called after disconnect delay)."""
    ws = comp.op('websocket1')

    if not ws:
        return

    # Set WebSocket netaddress and port parameters to expressions that reference parent parameters
    # This makes them update automatically when parent parameters change
    ws.par.netaddress.expr = 'f"ws://{parent().par.Host}/stream?window_size={parent().par.Windowsize}&overlap={parent().par.Overlap}&max_fps={parent().par.Maxfps}&quality={parent().par.Quality}"'
    ws.par.netaddress.mode = ParMode.EXPRESSION

    ws.par.port.expr = 'parent().par.Port'
    ws.par.port.mode = ParMode.EXPRESSION

    ws.par.autoreconnect = True
    ws.par.active = True

    comp.par.Connectionstatus = "Connecting..."
    debug("Reconnecting with new settings...")


def restart_backend(comp):
    """
    Restart the streaming backend server.
    """
    import subprocess
    import os

    comp.par.Backendstatus = "Restarting..."

    # Stop existing backend
    stop_backend(comp)

    # Wait a bit
    import time
    time.sleep(1.0)

    # Build command
    backend_path = comp.par.Backendpath.eval()
    model_idx = comp.par.Model.eval()
    device_idx = comp.par.Device.eval()
    port = comp.par.Backendport.eval()
    process_res_idx = comp.par.Processres.eval()

    # Map menu indices to values
    model_names = ['DA3-SMALL', 'DA3-BASE', 'DA3-LARGE', 'DA3-GIANT', 'DA3NESTED-GIANT-LARGE']
    device_names = ['mps', 'cuda', 'cpu']
    process_res_values = [252, 378, 504, 756, 1008]

    model_dir_map = {
        'DA3-SMALL': 'depth-anything/DA3-SMALL',
        'DA3-BASE': 'depth-anything/DA3-BASE',
        'DA3-LARGE': 'depth-anything/DA3-LARGE',
        'DA3-GIANT': 'depth-anything/DA3-GIANT',
        'DA3NESTED-GIANT-LARGE': 'depth-anything/DA3NESTED-GIANT-LARGE'
    }

    model_name = model_names[model_idx] if 0 <= model_idx < len(model_names) else 'DA3-SMALL'
    device = device_names[device_idx] if 0 <= device_idx < len(device_names) else 'mps'
    process_res = process_res_values[process_res_idx] if 0 <= process_res_idx < len(process_res_values) else 252

    model_dir = model_dir_map.get(model_name, comp.par.Modeldir.eval())

    # Activate venv and run da3 stream
    activate_script = os.path.join(backend_path, '.venv', 'bin', 'activate')
    cmd = f'source "{activate_script}" && cd "{backend_path}" && da3 stream --model-dir {model_dir} --device {device} --port {port} --process-res {process_res} --host 0.0.0.0'

    debug(f"Starting backend: {cmd}")

    try:
        # Store process handle in storage
        proc = subprocess.Popen(
            cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # Create new process group
        )

        comp.store('da3_backend_process', proc)
        comp.par.Backendstatus = f"Running (PID: {proc.pid})"
        debug(f"Backend started (PID: {proc.pid})")

    except Exception as e:
        comp.par.Backendstatus = f"Error: {str(e)}"
        debug(f"ERROR: Failed to start backend: {e}")


def stop_backend(comp):
    """
    Stop the streaming backend server.
    """
    import signal
    import os

    proc = comp.fetch('da3_backend_process', None)

    if proc and proc.poll() is None:  # Process is running
        try:
            # Kill process group
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            debug(f"Stopped backend (PID: {proc.pid})")
            comp.par.Backendstatus = "Stopped"
        except Exception as e:
            debug(f"ERROR: Error stopping backend: {e}")
            comp.par.Backendstatus = f"Error: {str(e)}"
    else:
        comp.par.Backendstatus = "Not running"
        debug("WARNING: Backend not running")


def check_backend_health(comp):
    """
    Check if backend is responsive.

    Returns:
        bool: True if healthy
    """
    import urllib.request
    import json

    host = comp.par.Host.eval()
    port = comp.par.Port.eval()

    try:
        url = f"http://{host}:{port}/health"
        with urllib.request.urlopen(url, timeout=2) as response:
            data = json.loads(response.read())
            if data.get('status') == 'healthy':
                comp.par.Backendstatus = f"Healthy ({data.get('model', 'unknown')})"
                return True
    except Exception as e:
        debug(f"WARNING: Backend health check failed: {e}")
        comp.par.Backendstatus = "Unreachable"

    return False


# WebSocket Callbacks

def onOpen(dat):
    """Called when WebSocket connection opens."""
    comp = dat.parent()
    comp.par.Connectionstatus = "Connected"
    debug("WebSocket connected!")


def onClose(dat):
    """Called when WebSocket connection closes."""
    comp = dat.parent()
    comp.par.Connectionstatus = "Disconnected"
    debug("WebSocket disconnected")


def onReceiveText(dat, rowIndex, message):
    """
    Called when text/JSON message is received from server.

    Args:
        dat: The WebSocket DAT
        rowIndex: Row index (usually 0)
        message: The text message received
    """
    import json
    import base64
    import numpy as np

    comp = dat.parent()

    try:
        data = json.loads(message)

        if data.get('status') == 'success':
            # Get format
            data_format = data.get('format', 'jpeg_base64')
            depth_b64 = data['data']
            shape = tuple(data['shape'])
            depth_min = data.get('min', 0.0)
            depth_max = data.get('max', 1.0)
            seq_id = data.get('seq_id', None)

            # Decode based on format
            if data_format == 'raw_float32':
                # Raw float32 numpy array
                depth_bytes = base64.b64decode(depth_b64)
                depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(shape)
                debug(f"Received raw float32 depth: {shape}, range={depth_min:.3f}-{depth_max:.3f}, seq={seq_id}")
            else:
                # JPEG uint8 - need to denormalize
                depth_bytes = base64.b64decode(depth_b64)
                depth_uint8 = np.frombuffer(depth_bytes, dtype=np.uint8).reshape(shape)
                # Denormalize from 0-255 back to original range
                depth = (depth_uint8.astype(np.float32) / 255.0) * (depth_max - depth_min) + depth_min
                debug(f"Received JPEG depth: {shape}, denormalized to range={depth.min():.3f}-{depth.max():.3f}, seq={seq_id}")

            # Store depth array in comp storage
            comp.store('depth_array', depth)
            comp.store('depth_min', depth_min)
            comp.store('depth_max', depth_max)

            # Retrieve matching RGB frame for synchronization
            if seq_id is not None:
                sync_dat = comp.op('da3_frame_sync')
                if sync_dat:
                    sync_mgr = sync_dat.module.get_sync_manager(comp)
                    synced_rgb = sync_mgr.get_frame_for_depth(seq_id)
                    if synced_rgb is not None:
                        comp.store('synced_rgb_frame', synced_rgb)
                        debug(f"Retrieved synced RGB frame for seq={seq_id}")
                    else:
                        debug(f"WARNING: No RGB frame found for seq={seq_id} (aged out?)")

            # Trigger depth display Script TOP to cook
            depth_display = comp.op('depth_display')
            if depth_display:
                depth_display.cook(force=True)

            # Trigger synced RGB output Script TOP to cook
            synced_output = comp.op('out_synched')
            if synced_output:
                synced_output.cook(force=True)

            # Update status with stats
            stats = data.get('stats', {})
            fps = stats.get('fps', 0)
            comp.par.Connectionstatus = f"Streaming @ {fps:.1f} FPS"
            comp.par.Currentfps = fps

            # Update received frame count
            comp.par.Framesreceived = comp.par.Framesreceived.eval() + 1

        elif data.get('status') == 'buffering':
            stats = data.get('stats', {})
            buffer_fill = stats.get('frames_in_buffer', 0)
            comp.par.Connectionstatus = f"Buffering ({buffer_fill} frames)"

        elif data.get('status') == 'error':
            error_msg = data.get('message', 'Unknown error')
            comp.par.Connectionstatus = f"Error: {error_msg}"
            debug(f"ERROR: Server error: {error_msg}")

    except Exception as e:
        debug(f"ERROR: Error processing message: {e}")
        import traceback
        traceback.print_exc()


def onReceiveBinary(dat, bytes):
    """Called when binary message is received (unexpected)."""
    debug(f"Received unexpected binary: {len(bytes)} bytes")


# Utility

def debug(msg):
    """Print debug message with timestamp."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] DA3 Stream: {msg}")
