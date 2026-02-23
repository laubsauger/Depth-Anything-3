# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sliding Window Processor for handling infinite-length sequences.

This module provides utilities for processing long sequences of images using
overlapping windows, enabling real-time streaming and infinite-length processing.
"""

from typing import List, Optional, Tuple, Iterator
import numpy as np
from dataclasses import dataclass


@dataclass
class WindowConfig:
    """Configuration for sliding window processing."""

    window_size: int = 12  # Number of frames per window
    overlap: int = 4  # Number of overlapping frames between windows
    blend_frames: int = 2  # Number of frames to blend at window boundaries

    def __post_init__(self):
        """Validate configuration."""
        assert self.window_size > 0, "window_size must be positive"
        assert 0 <= self.overlap < self.window_size, "overlap must be in [0, window_size)"
        assert 0 <= self.blend_frames <= self.overlap, "blend_frames must be <= overlap"

    @property
    def stride(self) -> int:
        """Step size between windows."""
        return self.window_size - self.overlap

    @classmethod
    def for_device(cls, device: str) -> "WindowConfig":
        """Create optimal config for device."""
        if device == "mps":
            return cls(window_size=12, overlap=4, blend_frames=2)
        elif device == "cuda":
            return cls(window_size=24, overlap=8, blend_frames=4)
        else:  # CPU
            return cls(window_size=8, overlap=3, blend_frames=1)


class SlidingWindowProcessor:
    """
    Process long sequences using overlapping sliding windows.

    This enables processing of arbitrarily long sequences by breaking them into
    manageable chunks with overlap to maintain temporal consistency.

    Example:
        >>> config = WindowConfig.for_device("mps")
        >>> processor = SlidingWindowProcessor(config)
        >>>
        >>> # Process 100 images in windows of 12 with overlap of 4
        >>> for window_idx, image_indices in processor.get_windows(100):
        >>>     # Process images[image_indices]
        >>>     # Blend results in overlap regions
        """

    def __init__(self, config: WindowConfig):
        self.config = config

    def get_windows(self, num_items: int) -> Iterator[Tuple[int, List[int]]]:
        """
        Generate sliding windows over sequence.

        Args:
            num_items: Total number of items in sequence

        Yields:
            Tuple of (window_index, list_of_item_indices)
        """
        if num_items <= self.config.window_size:
            # Single window covers everything
            yield (0, list(range(num_items)))
            return

        window_idx = 0
        start_idx = 0

        while start_idx < num_items:
            end_idx = min(start_idx + self.config.window_size, num_items)
            indices = list(range(start_idx, end_idx))

            yield (window_idx, indices)

            # Move to next window
            start_idx += self.config.stride
            window_idx += 1

            # Stop if we've covered all items
            if end_idx >= num_items:
                break

    def blend_overlaps(
        self,
        results: List[np.ndarray],
        window_indices: List[List[int]],
    ) -> np.ndarray:
        """
        Blend overlapping regions from multiple windows.

        Uses linear interpolation in the blend zone to smooth transitions
        between windows.

        Args:
            results: List of result arrays from each window
            window_indices: List of index lists for each window

        Returns:
            Blended array covering all frames
        """
        if len(results) == 1:
            return results[0]

        # Determine output shape
        total_frames = max(max(indices) for indices in window_indices) + 1
        result_shape = results[0].shape[1:]  # (H, W) or (H, W, C)

        # Initialize output and weight arrays
        output = np.zeros((total_frames, *result_shape), dtype=np.float32)
        weights = np.zeros((total_frames, 1), dtype=np.float32)

        # Blend each window's contribution
        for window_result, indices in zip(results, window_indices):
            window_size = len(indices)

            # Create blend weights (1.0 in center, fade at edges)
            window_weights = np.ones((window_size, 1), dtype=np.float32)

            # Apply fade-in at start
            if self.config.blend_frames > 0:
                fade_in = np.linspace(0, 1, self.config.blend_frames)[:, None]
                window_weights[:self.config.blend_frames] = fade_in

            # Apply fade-out at end
            if self.config.blend_frames > 0:
                fade_out = np.linspace(1, 0, self.config.blend_frames)[:, None]
                window_weights[-self.config.blend_frames:] = fade_out

            # Accumulate weighted results
            for i, frame_idx in enumerate(indices):
                weight = window_weights[i]
                output[frame_idx] += window_result[i] * weight
                weights[frame_idx] += weight

        # Normalize by total weight
        # Avoid division by zero
        weights = np.maximum(weights, 1e-8)
        output = output / weights

        return output

    def estimate_num_windows(self, num_items: int) -> int:
        """Estimate number of windows needed for sequence."""
        if num_items <= self.config.window_size:
            return 1
        return int(np.ceil((num_items - self.config.window_size) / self.config.stride)) + 1


def process_with_sliding_windows(
    items: List,
    process_fn,
    device: str = "mps",
    config: Optional[WindowConfig] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, list]:
    """
    Process a long sequence using sliding windows.

    Args:
        items: List of items to process (e.g., image paths)
        process_fn: Function that takes a list of items and returns results
        device: Device being used (for optimal config)
        config: Optional custom WindowConfig
        verbose: Whether to print progress

    Returns:
        Tuple of (blended_results, list_of_window_results)
    """
    if config is None:
        config = WindowConfig.for_device(device)

    processor = SlidingWindowProcessor(config)
    num_windows = processor.estimate_num_windows(len(items))

    if verbose:
        print(f"🪟 Processing {len(items)} items in {num_windows} windows")
        print(f"   Window size: {config.window_size}, Overlap: {config.overlap}, Blend: {config.blend_frames}")

    window_results = []
    window_indices_list = []

    for window_idx, indices in processor.get_windows(len(items)):
        if verbose:
            print(f"   Window {window_idx + 1}/{num_windows}: frames {indices[0]}-{indices[-1]}")

        # Get items for this window
        window_items = [items[i] for i in indices]

        # Process window
        result = process_fn(window_items)

        window_results.append(result)
        window_indices_list.append(indices)

    # Blend overlapping regions
    if verbose:
        print(f"🔀 Blending {len(window_results)} windows...")

    blended = processor.blend_overlaps(window_results, window_indices_list)

    if verbose:
        print(f"✅ Completed processing {len(items)} items")

    return blended, window_results
