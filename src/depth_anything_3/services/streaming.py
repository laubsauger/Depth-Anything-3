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
Streaming Depth Estimation Service

Provides real-time streaming depth estimation with sliding window processing.
Designed for integration with creative coding tools like TouchDesigner, vvvv, etc.
"""

from typing import List, Optional, Iterator, Tuple
import numpy as np
from collections import deque
from dataclasses import dataclass
import time

from ..api import DepthAnything3
from ..utils.sliding_window import WindowConfig
from ..specs import Prediction
from ..utils.timing import TimingBreakdown, AggregatedTimingStats


@dataclass
class StreamConfig:
    """Configuration for streaming processing."""

    buffer_size: int = 12  # Number of frames to buffer before processing
    overlap: int = 4  # Overlap with previous window
    output_latency_frames: int = 2  # Frames to wait before outputting (for blending)
    max_queue_size: int = 100  # Maximum frames to queue

    @classmethod
    def for_device(cls, device: str, latency: str = "balanced") -> "StreamConfig":
        """
        Create streaming config optimized for device and latency requirements.

        Args:
            device: "mps", "cuda", or "cpu"
            latency: "low" (faster, less accurate), "balanced", "high" (slower, more accurate)
        """
        if device == "mps":
            base_buffer = 12
        elif device == "cuda":
            base_buffer = 24
        else:  # CPU
            base_buffer = 8

        if latency == "low":
            return cls(
                buffer_size=base_buffer // 2,
                overlap=1,
                output_latency_frames=0,
            )
        elif latency == "high":
            return cls(
                buffer_size=base_buffer,
                overlap=base_buffer // 2,
                output_latency_frames=4,
            )
        else:  # balanced
            return cls(
                buffer_size=base_buffer,
                overlap=base_buffer // 3,
                output_latency_frames=2,
            )


class StreamingDepthEstimator:
    """
    Real-time streaming depth estimator using sliding windows.

    Example:
        >>> model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE").to("mps")
        >>> stream = StreamingDepthEstimator(model, device="mps")
        >>>
        >>> # Process frames one at a time
        >>> for frame in video_frames:
        >>>     depth = stream.process_frame(frame)
        >>>     if depth is not None:
        >>>         # Output is available (may be from a previous frame due to buffering)
        >>>         send_to_touchdesigner(depth)
        >>>
        >>> # Flush remaining frames
        >>> for depth in stream.flush():
        >>>     send_to_touchdesigner(depth)
    """

    def __init__(
        self,
        model: DepthAnything3,
        device: str = "mps",
        config: Optional[StreamConfig] = None,
        process_res: int = 504,
        collect_timing: bool = False,
    ):
        self.model = model
        self.device = device
        self.config = config or StreamConfig.for_device(device)
        self.process_res = process_res
        self.collect_timing = collect_timing

        # Sliding window state
        self.frame_buffer = deque(maxlen=self.config.buffer_size + self.config.overlap)
        self.output_buffer = deque(maxlen=self.config.max_queue_size)
        self.processed_count = 0
        self.frame_count = 0

        # Performance tracking
        self.last_process_time = 0.0
        self.avg_fps = 0.0

        # Timing tracking
        self.timing_stats = AggregatedTimingStats() if collect_timing else None

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Process a single frame.

        Args:
            frame: Input frame as numpy array (H, W, 3) or path string

        Returns:
            Depth map if available, None if still buffering
        """
        self.frame_count += 1
        self.frame_buffer.append(frame)

        # Process when buffer is full
        if len(self.frame_buffer) >= self.config.buffer_size:
            self._process_buffer()

        # Return output if available
        if len(self.output_buffer) > self.config.output_latency_frames:
            return self.output_buffer.popleft()

        return None

    def _process_buffer(self):
        """Process current buffer and add results to output queue."""
        if len(self.frame_buffer) == 0:
            return

        start_time = time.time()

        # Convert buffer to list for processing
        frames = list(self.frame_buffer)

        # Run inference
        prediction = self.model.inference(
            image=frames,
            export_dir=None,  # No export, just get results
            process_res=self.process_res,
            process_res_method="upper_bound_resize",
            collect_timing=self.collect_timing,
        )

        # Extract depth maps
        depth_maps = prediction.depth  # (N, H, W)

        # Add to output buffer (skip overlap frames to avoid duplicates)
        skip_frames = self.config.overlap if self.processed_count > 0 else 0
        for i in range(skip_frames, len(depth_maps)):
            self.output_buffer.append(depth_maps[i])

        # Keep overlap frames for next window
        overlap_frames = list(self.frame_buffer)[-self.config.overlap:]
        self.frame_buffer.clear()
        self.frame_buffer.extend(overlap_frames)

        self.processed_count += 1

        # Update FPS tracking
        process_time = time.time() - start_time
        self.last_process_time = process_time
        window_fps = len(frames) / process_time if process_time > 0 else 0
        self.avg_fps = (self.avg_fps * 0.9 + window_fps * 0.1) if self.avg_fps > 0 else window_fps

        # Track timing if enabled
        if self.collect_timing and hasattr(prediction, 'timing'):
            self.timing_stats.add_breakdown(prediction.timing)

    def flush(self) -> Iterator[np.ndarray]:
        """
        Flush remaining frames in buffer.

        Yields:
            Remaining depth maps
        """
        # Process any remaining frames
        if len(self.frame_buffer) > 0:
            self._process_buffer()

        # Output all buffered results
        while self.output_buffer:
            yield self.output_buffer.popleft()

    def get_stats(self) -> dict:
        """Get streaming statistics."""
        stats = {
            "frames_received": self.frame_count,
            "frames_processed": self.processed_count * self.config.buffer_size,
            "frames_in_buffer": len(self.frame_buffer),
            "frames_in_output_queue": len(self.output_buffer),
            "last_process_time_ms": self.last_process_time * 1000,
            "avg_fps": self.avg_fps,
            "latency_frames": self.config.output_latency_frames,
        }

        # Add timing stats if available
        if self.timing_stats is not None:
            stats["timing"] = self.timing_stats.to_dict()

        return stats

    def reset(self):
        """Reset streaming state."""
        self.frame_buffer.clear()
        self.output_buffer.clear()
        self.processed_count = 0
        self.frame_count = 0
        self.last_process_time = 0.0
        self.avg_fps = 0.0
