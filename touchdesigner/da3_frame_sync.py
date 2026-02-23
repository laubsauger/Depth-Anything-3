"""
Frame Synchronization System for DA3 Streaming

Manages synchronization between sent RGB frames and received depth maps.
Uses sequence numbers and a circular buffer to match frames.

Usage:
1. Called from send_frames to store RGB frames with sequence IDs
2. Called from stream_control when depth arrives to retrieve matching RGB
3. Outputs synced RGB frame to Script TOP
"""

import numpy as np
from collections import deque


class FrameSyncManager:
    """
    Manages frame synchronization between RGB input and depth output.
    """

    def __init__(self, buffer_size=30):
        """
        Initialize frame sync manager.

        Args:
            buffer_size: Maximum frames to keep in buffer (default 30)
        """
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        self.next_seq_id = 0
        self.last_synced_seq = None
        self.last_synced_frame = None

    def store_sent_frame(self, frame_array, timestamp=None):
        """
        Store a frame that was sent to the server.

        Args:
            frame_array: numpy array (H, W, C) of RGB frame
            timestamp: optional timestamp

        Returns:
            int: Sequence ID for this frame
        """
        seq_id = self.next_seq_id
        self.next_seq_id += 1

        # Store frame with metadata
        self.frame_buffer.append({
            'seq_id': seq_id,
            'frame': frame_array.copy(),  # Copy to avoid reference issues
            'timestamp': timestamp
        })

        return seq_id

    def get_frame_for_depth(self, seq_id):
        """
        Retrieve the RGB frame matching a depth sequence ID.

        Args:
            seq_id: Sequence ID from depth response

        Returns:
            numpy array or None: RGB frame matching this depth, or None if not found
        """
        # Search buffer for matching sequence ID
        for frame_data in self.frame_buffer:
            if frame_data['seq_id'] == seq_id:
                self.last_synced_seq = seq_id
                self.last_synced_frame = frame_data['frame']
                return frame_data['frame']

        # Not found - frame may have aged out of buffer
        # Return last synced frame as fallback
        return self.last_synced_frame

    def get_latest_frame(self):
        """
        Get the most recently stored frame (for out_input).

        Returns:
            numpy array or None: Latest RGB frame
        """
        if self.frame_buffer:
            return self.frame_buffer[-1]['frame']
        return None

    def get_synced_frame(self):
        """
        Get the last synced frame (for out_synched).

        Returns:
            numpy array or None: Last synced RGB frame
        """
        return self.last_synced_frame

    def clear(self):
        """Clear the frame buffer."""
        self.frame_buffer.clear()
        self.next_seq_id = 0
        self.last_synced_seq = None
        self.last_synced_frame = None


def get_sync_manager(comp):
    """
    Get or create the FrameSyncManager for this comp.

    Args:
        comp: Parent COMP

    Returns:
        FrameSyncManager: The sync manager instance
    """
    sync_mgr = comp.fetch('frame_sync_manager', None)

    if sync_mgr is None:
        # Create new sync manager with buffer size from parameter
        buffer_size = 30  # Default
        try:
            buffer_size = int(comp.par.Syncbuffersize.eval())
        except:
            pass

        sync_mgr = FrameSyncManager(buffer_size=buffer_size)
        comp.store('frame_sync_manager', sync_mgr)

    return sync_mgr


def debug(msg):
    """Print debug message with timestamp."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] Frame Sync: {msg}")
