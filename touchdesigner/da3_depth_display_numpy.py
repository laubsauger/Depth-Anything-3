"""
Depth Map Display Script TOP (Numpy Version)

Displays depth maps received as numpy arrays from parent storage.
Reads ALL parameters from parent COMP.
No PIL required - uses only TouchDesigner built-in numpy support.

Usage:
1. Create a Script TOP named 'depth_display' inside parent COMP
2. Paste this code
3. The script will automatically fetch depth data and settings from parent
"""

import numpy as np


def onCook(scriptOp):
    """
    Called when the Script TOP needs to update.
    Fetches depth array from storage and renders it.
    """
    # Get parent COMP
    comp = scriptOp.parent()

    # Get the latest depth array from parent storage
    depth_array = comp.fetch('depth_array', None)

    if depth_array is None:
        # No data yet, create black image
        black = np.zeros((256, 256, 4), dtype=np.float32)
        scriptOp.copyNumpyArray(black)
        return

    try:
        # Make a copy to avoid modifying the original
        depth = depth_array.copy()

        # Ensure 2D
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]

        # Normalize if requested (read from parent parameter)
        if comp.par.Normalize.eval():
            depth_min = depth.min()
            depth_max = depth.max()
            if depth_max > depth_min:
                depth = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth = np.zeros_like(depth)
        else:
            # Use stored min/max for consistent normalization
            depth_min = comp.fetch('depth_min', depth.min())
            depth_max = comp.fetch('depth_max', depth.max())
            if depth_max > depth_min:
                depth = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth = np.zeros_like(depth)

        # Apply invert (read from parent parameter)
        if comp.par.Invert.eval():
            depth = 1.0 - depth

        # Apply brightness and contrast (read from parent parameters)
        brightness = comp.par.Brightness.eval()
        contrast = comp.par.Contrast.eval()
        depth = (depth - 0.5) * contrast + 0.5
        depth = depth * brightness
        depth = np.clip(depth, 0.0, 1.0)

        # Apply colormap if enabled (read from parent parameters)
        if comp.par.Colorize.eval():
            colormap_name = comp.par.Colormap.eval()
            depth_rgb = apply_colormap(depth, colormap_name)
        else:
            # Grayscale - stack to RGB
            depth_rgb = np.stack([depth] * 3, axis=-1)

        # TouchDesigner Script TOP expects (height, width, channels) in float32 [0, 1]
        # Add alpha channel (RGBA)
        h, w = depth_rgb.shape[0], depth_rgb.shape[1]
        depth_rgba = np.ones((h, w, 4), dtype=np.float32)
        depth_rgba[:, :, :3] = depth_rgb.astype(np.float32)

        # Copy to TOP using copyNumpyArray
        scriptOp.copyNumpyArray(depth_rgba)

    except Exception as e:
        print(f"ERROR: Error in Script TOP: {e}")
        import traceback
        traceback.print_exc()
        # Show error as red screen
        error_img = np.zeros((256, 256, 4), dtype=np.float32)
        error_img[:, :, 0] = 0.5  # Red channel
        error_img[:, :, 3] = 1.0  # Alpha
        scriptOp.copyNumpyArray(error_img)


def apply_colormap(depth, colormap_name):
    """
    Apply a colormap to depth values.

    Args:
        depth: (H, W) array of depth values in [0, 1]
        colormap_name: Name of colormap

    Returns:
        (H, W, 3) RGB array in [0, 1]
    """
    if colormap_name == 'gray':
        return np.stack([depth] * 3, axis=-1)

    elif colormap_name == 'viridis':
        # Approximate viridis colormap
        r = np.interp(depth, [0, 0.25, 0.5, 0.75, 1.0],
                      [0.267, 0.229, 0.127, 0.369, 0.993])
        g = np.interp(depth, [0, 0.25, 0.5, 0.75, 1.0],
                      [0.005, 0.322, 0.566, 0.788, 0.906])
        b = np.interp(depth, [0, 0.25, 0.5, 0.75, 1.0],
                      [0.329, 0.545, 0.551, 0.382, 0.144])
        return np.stack([r, g, b], axis=-1)

    elif colormap_name == 'plasma':
        # Approximate plasma colormap
        r = np.interp(depth, [0, 0.25, 0.5, 0.75, 1.0],
                      [0.050, 0.522, 0.826, 0.958, 0.940])
        g = np.interp(depth, [0, 0.25, 0.5, 0.75, 1.0],
                      [0.030, 0.109, 0.329, 0.647, 0.975])
        b = np.interp(depth, [0, 0.25, 0.5, 0.75, 1.0],
                      [0.529, 0.639, 0.570, 0.273, 0.131])
        return np.stack([r, g, b], axis=-1)

    elif colormap_name == 'inferno':
        # Approximate inferno colormap
        r = np.interp(depth, [0, 0.25, 0.5, 0.75, 1.0],
                      [0.001, 0.258, 0.651, 0.933, 0.988])
        g = np.interp(depth, [0, 0.25, 0.5, 0.75, 1.0],
                      [0.000, 0.048, 0.234, 0.565, 0.998])
        b = np.interp(depth, [0, 0.25, 0.5, 0.75, 1.0],
                      [0.014, 0.239, 0.100, 0.052, 0.645])
        return np.stack([r, g, b], axis=-1)

    elif colormap_name == 'magma':
        # Approximate magma colormap
        r = np.interp(depth, [0, 0.25, 0.5, 0.75, 1.0],
                      [0.001, 0.302, 0.716, 0.950, 0.987])
        g = np.interp(depth, [0, 0.25, 0.5, 0.75, 1.0],
                      [0.000, 0.064, 0.215, 0.517, 0.991])
        b = np.interp(depth, [0, 0.25, 0.5, 0.75, 1.0],
                      [0.014, 0.430, 0.524, 0.571, 0.749])
        return np.stack([r, g, b], axis=-1)

    elif colormap_name == 'turbo':
        # Approximate turbo (smooth rainbow)
        r = np.interp(depth, [0, 0.25, 0.5, 0.75, 1.0],
                      [0.190, 0.020, 0.450, 0.960, 0.600])
        g = np.interp(depth, [0, 0.25, 0.5, 0.75, 1.0],
                      [0.070, 0.560, 0.950, 0.820, 0.080])
        b = np.interp(depth, [0, 0.25, 0.5, 0.75, 1.0],
                      [0.480, 0.920, 0.460, 0.050, 0.020])
        return np.stack([r, g, b], axis=-1)

    else:
        # Default to grayscale
        return np.stack([depth] * 3, axis=-1)
