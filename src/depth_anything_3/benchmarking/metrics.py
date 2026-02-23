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
Metrics Collection System

Collects and tracks performance metrics during benchmarking.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time
import numpy as np
import psutil
import torch

from ..utils.timing import TimingBreakdown, AggregatedTimingStats


@dataclass
class FrameMetrics:
    """Metrics for a single frame processing."""

    frame_idx: int
    inference_time_ms: float  # Time for model inference
    total_time_ms: float  # Total time including pre/post processing
    memory_used_mb: float  # GPU/MPS memory used
    memory_allocated_mb: float  # GPU/MPS memory allocated
    timing_breakdown: Optional[TimingBreakdown] = None  # Detailed timing breakdown


@dataclass
class BenchmarkMetrics:
    """Complete metrics for a benchmark run."""

    scenario_name: str
    device: str

    # Frame-level metrics
    frame_metrics: List[FrameMetrics] = field(default_factory=list)

    # Aggregate statistics
    total_frames: int = 0
    total_time_s: float = 0.0

    # Performance metrics
    avg_fps: float = 0.0
    min_fps: float = 0.0
    max_fps: float = 0.0
    std_fps: float = 0.0

    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Memory metrics
    avg_memory_mb: float = 0.0
    max_memory_mb: float = 0.0
    peak_allocated_mb: float = 0.0

    # Temporal consistency metrics
    temporal_jitter: float = 0.0  # Std dev of frame-to-frame depth changes
    temporal_smoothness: float = 0.0  # 1 - (avg abs diff between consecutive frames)
    flicker_score: float = 0.0  # High frequency changes (bad = high)

    # Pose estimation metrics
    pose_estimation_enabled: bool = False
    avg_pose_confidence: float = 0.0  # Average confidence if available
    pose_conditioned_enabled: bool = False

    # System metrics
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0

    # Detailed timing breakdown (aggregated)
    timing_stats: Optional[AggregatedTimingStats] = None

    def add_frame(self, frame_metric: FrameMetrics):
        """Add a frame metric."""
        self.frame_metrics.append(frame_metric)
        self.total_frames += 1

        # Collect timing breakdown if available
        if frame_metric.timing_breakdown is not None:
            if self.timing_stats is None:
                self.timing_stats = AggregatedTimingStats()
            self.timing_stats.add_breakdown(frame_metric.timing_breakdown)

    def compute_statistics(self):
        """Compute aggregate statistics from frame metrics."""
        if not self.frame_metrics:
            return

        # Extract latencies
        latencies = [fm.total_time_ms for fm in self.frame_metrics]
        inference_times = [fm.inference_time_ms for fm in self.frame_metrics]

        # Compute overall FPS from total time and frames
        # This is the true throughput, not per-frame instantaneous FPS
        if self.total_time_s > 0:
            self.avg_fps = self.total_frames / self.total_time_s
        else:
            self.avg_fps = 0.0

        # Min/max FPS are not meaningful in buffered processing, set to avg
        self.min_fps = self.avg_fps
        self.max_fps = self.avg_fps
        self.std_fps = 0.0

        # Latency percentiles
        self.avg_latency_ms = np.mean(latencies)
        self.p50_latency_ms = np.percentile(latencies, 50)
        self.p95_latency_ms = np.percentile(latencies, 95)
        self.p99_latency_ms = np.percentile(latencies, 99)

        # Memory statistics
        memory_values = [fm.memory_used_mb for fm in self.frame_metrics]
        allocated_values = [fm.memory_allocated_mb for fm in self.frame_metrics]

        self.avg_memory_mb = np.mean(memory_values)
        self.max_memory_mb = np.max(memory_values)
        self.peak_allocated_mb = np.max(allocated_values)

        # System metrics
        self.cpu_percent = psutil.cpu_percent(interval=None)
        self.ram_used_gb = psutil.virtual_memory().used / (1024**3)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        result = {
            "scenario_name": self.scenario_name,
            "device": self.device,
            "summary": {
                "total_frames": self.total_frames,
                "total_time_s": self.total_time_s,
                "avg_fps": round(self.avg_fps, 2),
                "min_fps": round(self.min_fps, 2),
                "max_fps": round(self.max_fps, 2),
                "std_fps": round(self.std_fps, 2),
            },
            "latency": {
                "avg_ms": round(self.avg_latency_ms, 2),
                "p50_ms": round(self.p50_latency_ms, 2),
                "p95_ms": round(self.p95_latency_ms, 2),
                "p99_ms": round(self.p99_latency_ms, 2),
            },
            "memory": {
                "avg_mb": round(self.avg_memory_mb, 2),
                "max_mb": round(self.max_memory_mb, 2),
                "peak_allocated_mb": round(self.peak_allocated_mb, 2),
            },
            "system": {
                "cpu_percent": round(self.cpu_percent, 2),
                "ram_used_gb": round(self.ram_used_gb, 2),
            },
            "frame_metrics": [
                {
                    "frame_idx": fm.frame_idx,
                    "inference_time_ms": round(fm.inference_time_ms, 2),
                    "total_time_ms": round(fm.total_time_ms, 2),
                    "memory_used_mb": round(fm.memory_used_mb, 2),
                }
                for fm in self.frame_metrics
            ],
        }

        # Add timing breakdown if available
        if self.timing_stats is not None:
            result["timing_breakdown"] = self.timing_stats.to_dict()

        return result

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"Benchmark Results: {self.scenario_name}\n"
            f"{'=' * 60}\n"
            f"Total Frames: {self.total_frames}\n"
            f"Total Time: {self.total_time_s:.2f}s\n"
            f"\n"
            f"Performance:\n"
            f"  Average FPS: {self.avg_fps:.2f} (min: {self.min_fps:.2f}, max: {self.max_fps:.2f})\n"
            f"  Latency: {self.avg_latency_ms:.2f}ms (p95: {self.p95_latency_ms:.2f}ms)\n"
            f"\n"
            f"Memory:\n"
            f"  Average: {self.avg_memory_mb:.2f} MB\n"
            f"  Peak: {self.max_memory_mb:.2f} MB\n"
            f"\n"
            f"System:\n"
            f"  CPU: {self.cpu_percent:.1f}%\n"
            f"  RAM: {self.ram_used_gb:.2f} GB\n"
        )


class MetricsCollector:
    """
    Collects metrics during benchmark execution.
    """

    def __init__(self, scenario_name: str, device: str):
        self.metrics = BenchmarkMetrics(scenario_name=scenario_name, device=device)
        self.current_frame_start = None
        self.current_inference_start = None

    def start_frame(self, frame_idx: int):
        """Mark the start of frame processing."""
        self.current_frame_idx = frame_idx
        self.current_frame_start = time.time()

    def start_inference(self):
        """Mark the start of model inference."""
        self.current_inference_start = time.time()

    def end_inference(self) -> float:
        """Mark the end of inference and return inference time."""
        if self.current_inference_start is None:
            return 0.0
        inference_time = (time.time() - self.current_inference_start) * 1000
        self.current_inference_time = inference_time
        return inference_time

    def end_frame(self):
        """Mark the end of frame processing and record metrics."""
        if self.current_frame_start is None:
            return

        total_time_ms = (time.time() - self.current_frame_start) * 1000

        # Get memory usage
        memory_used_mb, memory_allocated_mb = self._get_memory_usage()

        # Create frame metric
        frame_metric = FrameMetrics(
            frame_idx=self.current_frame_idx,
            inference_time_ms=getattr(self, 'current_inference_time', total_time_ms),
            total_time_ms=total_time_ms,
            memory_used_mb=memory_used_mb,
            memory_allocated_mb=memory_allocated_mb,
        )

        self.metrics.add_frame(frame_metric)

        # Reset
        self.current_frame_start = None
        self.current_inference_start = None

    def _get_memory_usage(self) -> tuple[float, float]:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_reserved() / (1024**2)
            memory_allocated = torch.cuda.memory_allocated() / (1024**2)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't have direct memory tracking like CUDA
            # Use allocated memory as approximation
            memory_used = torch.mps.current_allocated_memory() / (1024**2)
            memory_allocated = memory_used
        else:
            memory_used = 0.0
            memory_allocated = 0.0

        return memory_used, memory_allocated

    def finalize(self, total_time_s: float) -> BenchmarkMetrics:
        """Finalize metrics and compute statistics."""
        self.metrics.total_time_s = total_time_s
        self.metrics.compute_statistics()
        return self.metrics
