"""
Output: Synched RGB Frame (Script TOP)

Displays the RGB frame that corresponds to the current depth output.
This is synchronized with the depth map via sequence IDs.

Usage:
1. Create Script TOP named 'out_synched'
2. Set script to this DAT
"""

import numpy as np


def onCook(scriptOp):
    """
    Called when Script TOP needs to update.
    Outputs the RGB frame synchronized with the current depth frame.
    """
    comp = scriptOp.parent()

    # Get synced RGB frame from storage
    synced_frame = comp.fetch('synced_rgb_frame', None)

    if synced_frame is None:
        # No synced frame yet - output black
        black = np.zeros((256, 256, 4), dtype=np.float32)
        scriptOp.copyNumpyArray(black)
        return

    # Convert to RGBA format for Script TOP
    # synced_frame is (H, W, C) in float32 [0, 1]
    h, w = synced_frame.shape[:2]

    if synced_frame.shape[2] == 3:
        # RGB -> RGBA
        rgba = np.ones((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = synced_frame
    else:
        # Already RGBA
        rgba = synced_frame.astype(np.float32)

    scriptOp.copyNumpyArray(rgba)
