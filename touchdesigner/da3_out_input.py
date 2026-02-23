"""
Output: Latest Input RGB Frame (Script TOP)

Displays the most recently captured input frame.
This is the raw input before processing.

Usage:
1. Create Script TOP named 'out_input'
2. Set script to this DAT
"""

import numpy as np


def onCook(scriptOp):
    """
    Called when Script TOP needs to update.
    Outputs the latest captured input frame.
    """
    comp = scriptOp.parent()

    # Get sync manager
    sync_dat = comp.op('da3_frame_sync')
    if not sync_dat:
        # No sync manager - output black
        black = np.zeros((256, 256, 4), dtype=np.float32)
        scriptOp.copyNumpyArray(black)
        return

    sync_mgr = sync_dat.module.get_sync_manager(comp)
    latest_frame = sync_mgr.get_latest_frame()

    if latest_frame is None:
        # No frames yet - output black
        black = np.zeros((256, 256, 4), dtype=np.float32)
        scriptOp.copyNumpyArray(black)
        return

    # Convert to RGBA format for Script TOP
    # latest_frame is (H, W, C) in float32 [0, 1]
    h, w = latest_frame.shape[:2]

    if latest_frame.shape[2] == 3:
        # RGB -> RGBA
        rgba = np.ones((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = latest_frame
    else:
        # Already RGBA
        rgba = latest_frame.astype(np.float32)

    scriptOp.copyNumpyArray(rgba)
