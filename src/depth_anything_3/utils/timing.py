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
Detailed Timing Utilities

Provides fine-grained timing breakdown for benchmarking and profiling.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import time
import torch
import numpy as np


@dataclass
class TimingBreakdown:
    """
    Detailed timing breakdown for a single inference pass.

    All times are in milliseconds (ms).
    """

    # Input preprocessing stages
    image_loading_ms: float = 0.0  # Loading images from disk/memory
    image_resize_ms: float = 0.0  # Resizing to process resolution
    image_normalization_ms: float = 0.0  # ImageNet normalization
    camera_preprocessing_ms: float = 0.0  # Extrinsics/intrinsics processing
    total_preprocessing_ms: float = 0.0  # Total preprocessing time

    # Model input preparation
    to_device_ms: float = 0.0  # Moving tensors to GPU/MPS
    extrinsics_normalization_ms: float = 0.0  # Normalizing camera poses
    total_input_prep_ms: float = 0.0  # Total input preparation

    # Model forward pass (broken down further)
    model_forward_ms: float = 0.0  # Total model forward time
    backbone_forward_ms: float = 0.0  # DinoV2 backbone
    head_forward_ms: float = 0.0  # DPT head
    camera_encoder_ms: float = 0.0  # Camera encoder (if pose enabled)
    camera_decoder_ms: float = 0.0  # Camera decoder (if pose enabled)
    gs_adapter_ms: float = 0.0  # Gaussian Splatting adapter (if enabled)

    # Output processing stages
    output_to_cpu_ms: float = 0.0  # Moving results to CPU (includes aux features - LARGE for big models!)
    depth_postprocessing_ms: float = 0.0  # Sky segmentation, scaling, etc.
    pose_alignment_ms: float = 0.0  # Umeyama alignment
    prediction_conversion_ms: float = 0.0  # Converting to Prediction object (numpy conversion, negligible)
    total_postprocessing_ms: float = 0.0  # Total postprocessing time

    # Export (if enabled)
    export_ms: float = 0.0  # Export to disk (GLB, NPZ, etc.)

    # Overall timing
    total_ms: float = 0.0  # End-to-end time

    # Device synchronization overhead (CUDA only)
    sync_overhead_ms: float = 0.0

    def compute_totals(self):
        """Compute total times from sub-components."""
        self.total_preprocessing_ms = (
            self.image_loading_ms
            + self.image_resize_ms
            + self.image_normalization_ms
            + self.camera_preprocessing_ms
        )

        self.total_input_prep_ms = (
            self.to_device_ms
            + self.extrinsics_normalization_ms
        )

        self.total_postprocessing_ms = (
            self.output_to_cpu_ms
            + self.depth_postprocessing_ms
            + self.pose_alignment_ms
            + self.prediction_conversion_ms
        )

        self.total_ms = (
            self.total_preprocessing_ms
            + self.total_input_prep_ms
            + self.model_forward_ms
            + self.total_postprocessing_ms
            + self.export_ms
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "preprocessing": {
                "image_loading_ms": round(self.image_loading_ms, 2),
                "image_resize_ms": round(self.image_resize_ms, 2),
                "image_normalization_ms": round(self.image_normalization_ms, 2),
                "camera_preprocessing_ms": round(self.camera_preprocessing_ms, 2),
                "total_ms": round(self.total_preprocessing_ms, 2),
            },
            "input_preparation": {
                "to_device_ms": round(self.to_device_ms, 2),
                "extrinsics_normalization_ms": round(self.extrinsics_normalization_ms, 2),
                "total_ms": round(self.total_input_prep_ms, 2),
            },
            "model_forward": {
                "total_ms": round(self.model_forward_ms, 2),
                "backbone_forward_ms": round(self.backbone_forward_ms, 2),
                "head_forward_ms": round(self.head_forward_ms, 2),
                "camera_encoder_ms": round(self.camera_encoder_ms, 2),
                "camera_decoder_ms": round(self.camera_decoder_ms, 2),
                "gs_adapter_ms": round(self.gs_adapter_ms, 2),
            },
            "postprocessing": {
                "output_to_cpu_ms": round(self.output_to_cpu_ms, 2),
                "depth_postprocessing_ms": round(self.depth_postprocessing_ms, 2),
                "pose_alignment_ms": round(self.pose_alignment_ms, 2),
                "prediction_conversion_ms": round(self.prediction_conversion_ms, 2),
                "total_ms": round(self.total_postprocessing_ms, 2),
            },
            "export_ms": round(self.export_ms, 2),
            "sync_overhead_ms": round(self.sync_overhead_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }

    def get_percentage_breakdown(self) -> Dict[str, float]:
        """Get percentage breakdown of where time is spent."""
        if self.total_ms == 0:
            return {}

        return {
            "preprocessing": (self.total_preprocessing_ms / self.total_ms) * 100,
            "input_preparation": (self.total_input_prep_ms / self.total_ms) * 100,
            "model_forward": (self.model_forward_ms / self.total_ms) * 100,
            "postprocessing": (self.total_postprocessing_ms / self.total_ms) * 100,
            "export": (self.export_ms / self.total_ms) * 100,
            "sync_overhead": (self.sync_overhead_ms / self.total_ms) * 100,
        }

    def __str__(self) -> str:
        """Human-readable timing breakdown."""
        percentages = self.get_percentage_breakdown()

        lines = [
            "Timing Breakdown:",
            "=" * 60,
            f"Total: {self.total_ms:.2f}ms",
            "",
            "Preprocessing: {:.2f}ms ({:.1f}%)".format(
                self.total_preprocessing_ms,
                percentages.get("preprocessing", 0)
            ),
            f"  ├─ Image Loading: {self.image_loading_ms:.2f}ms",
            f"  ├─ Image Resize: {self.image_resize_ms:.2f}ms",
            f"  ├─ Image Normalization: {self.image_normalization_ms:.2f}ms",
            f"  └─ Camera Preprocessing: {self.camera_preprocessing_ms:.2f}ms",
            "",
            "Input Preparation: {:.2f}ms ({:.1f}%)".format(
                self.total_input_prep_ms,
                percentages.get("input_preparation", 0)
            ),
            f"  ├─ To Device: {self.to_device_ms:.2f}ms",
            f"  └─ Extrinsics Normalization: {self.extrinsics_normalization_ms:.2f}ms",
            "",
            "Model Forward: {:.2f}ms ({:.1f}%)".format(
                self.model_forward_ms,
                percentages.get("model_forward", 0)
            ),
            f"  ├─ Backbone: {self.backbone_forward_ms:.2f}ms",
            f"  ├─ Head: {self.head_forward_ms:.2f}ms",
            f"  ├─ Camera Encoder: {self.camera_encoder_ms:.2f}ms",
            f"  ├─ Camera Decoder: {self.camera_decoder_ms:.2f}ms",
            f"  └─ GS Adapter: {self.gs_adapter_ms:.2f}ms",
            "",
            "Postprocessing: {:.2f}ms ({:.1f}%)".format(
                self.total_postprocessing_ms,
                percentages.get("postprocessing", 0)
            ),
            f"  ├─ Output to CPU: {self.output_to_cpu_ms:.2f}ms",
            f"  ├─ Depth Postprocessing: {self.depth_postprocessing_ms:.2f}ms",
            f"  ├─ Pose Alignment: {self.pose_alignment_ms:.2f}ms",
            f"  └─ Prediction Conversion: {self.prediction_conversion_ms:.2f}ms",
            "",
            "Export: {:.2f}ms ({:.1f}%)".format(
                self.export_ms,
                percentages.get("export", 0)
            ),
        ]

        if self.sync_overhead_ms > 0:
            lines.append(
                "Sync Overhead: {:.2f}ms ({:.1f}%)".format(
                    self.sync_overhead_ms,
                    percentages.get("sync_overhead", 0)
                )
            )

        return "\n".join(lines)


@dataclass
class AggregatedTimingStats:
    """Aggregated timing statistics across multiple frames."""

    num_frames: int = 0

    # Mean timings
    mean_breakdown: Optional[TimingBreakdown] = None

    # Min/max timings for each stage
    min_breakdown: Optional[TimingBreakdown] = None
    max_breakdown: Optional[TimingBreakdown] = None

    # Standard deviations
    std_breakdown: Optional[TimingBreakdown] = None

    # All individual breakdowns
    all_breakdowns: List[TimingBreakdown] = field(default_factory=list)

    def add_breakdown(self, breakdown: TimingBreakdown):
        """Add a timing breakdown."""
        self.all_breakdowns.append(breakdown)
        self.num_frames = len(self.all_breakdowns)
        self._compute_statistics()

    def _compute_statistics(self):
        """Compute aggregated statistics from all breakdowns."""
        if not self.all_breakdowns:
            return

        # Extract all values into arrays
        fields = [
            'image_loading_ms', 'image_resize_ms', 'image_normalization_ms',
            'camera_preprocessing_ms', 'total_preprocessing_ms',
            'to_device_ms', 'extrinsics_normalization_ms', 'total_input_prep_ms',
            'model_forward_ms', 'backbone_forward_ms', 'head_forward_ms',
            'camera_encoder_ms', 'camera_decoder_ms', 'gs_adapter_ms',
            'output_to_cpu_ms', 'depth_postprocessing_ms', 'pose_alignment_ms',
            'prediction_conversion_ms', 'total_postprocessing_ms',
            'export_ms', 'sync_overhead_ms', 'total_ms'
        ]

        arrays = {}
        for field in fields:
            arrays[field] = np.array([getattr(b, field) for b in self.all_breakdowns])

        # Compute mean
        self.mean_breakdown = TimingBreakdown()
        for field in fields:
            setattr(self.mean_breakdown, field, float(np.mean(arrays[field])))

        # Compute min
        self.min_breakdown = TimingBreakdown()
        for field in fields:
            setattr(self.min_breakdown, field, float(np.min(arrays[field])))

        # Compute max
        self.max_breakdown = TimingBreakdown()
        for field in fields:
            setattr(self.max_breakdown, field, float(np.max(arrays[field])))

        # Compute std
        self.std_breakdown = TimingBreakdown()
        for field in fields:
            setattr(self.std_breakdown, field, float(np.std(arrays[field])))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_frames": self.num_frames,
            "mean": self.mean_breakdown.to_dict() if self.mean_breakdown else {},
            "min": self.min_breakdown.to_dict() if self.min_breakdown else {},
            "max": self.max_breakdown.to_dict() if self.max_breakdown else {},
            "std": self.std_breakdown.to_dict() if self.std_breakdown else {},
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        if not self.mean_breakdown:
            return "No timing data"

        lines = [
            f"Aggregated Timing Statistics ({self.num_frames} frames)",
            "=" * 60,
            "",
            "MEAN TIMINGS:",
            str(self.mean_breakdown),
            "",
            "STANDARD DEVIATIONS:",
            "=" * 60,
        ]

        # Add std dev breakdown
        percentages = self.mean_breakdown.get_percentage_breakdown()
        lines.extend([
            f"Preprocessing: ±{self.std_breakdown.total_preprocessing_ms:.2f}ms",
            f"Input Preparation: ±{self.std_breakdown.total_input_prep_ms:.2f}ms",
            f"Model Forward: ±{self.std_breakdown.model_forward_ms:.2f}ms",
            f"Postprocessing: ±{self.std_breakdown.total_postprocessing_ms:.2f}ms",
            f"Total: ±{self.std_breakdown.total_ms:.2f}ms",
        ])

        return "\n".join(lines)


class Timer:
    """
    Context manager for timing code blocks with device synchronization.

    Example:
        >>> timer = Timer(device='cuda')
        >>> with timer:
        >>>     result = model(input)
        >>> print(f"Elapsed: {timer.elapsed_ms:.2f}ms")
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = 0.0

    def __enter__(self):
        """Start timing."""
        self._sync()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing."""
        self._sync()
        self.end_time = time.time()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000

    def _sync(self):
        """Synchronize device if needed."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif self.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't have explicit synchronization, but we can do a dummy operation
            if torch.backends.mps.is_available():
                torch.mps.synchronize()