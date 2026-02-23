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
Benchmark Configuration System

Defines configuration for streaming benchmarks including model variants,
resolutions, precisions, and multi-camera setups.

Note on fp16 precision:
    fp16 is only supported on CUDA devices. On MPS or CPU, the benchmark
    runner will automatically fall back to fp32 with a warning.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Configuration for a specific model variant."""

    name: str  # e.g., "DA3-SMALL", "DA3-BASE", "DA3-LARGE"
    model_dir: str  # HuggingFace repo or local path
    description: str = ""

    def __post_init__(self):
        if not self.description:
            self.description = f"{self.name} model"


@dataclass
class StreamConfig:
    """Configuration for streaming parameters."""

    window_size: Optional[int] = None  # None = auto-detect
    overlap: Optional[int] = None
    buffer_size: Optional[int] = None
    output_latency_frames: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        return {
            "window_size": self.window_size,
            "overlap": self.overlap,
            "buffer_size": self.buffer_size,
            "output_latency_frames": self.output_latency_frames,
        }


@dataclass
class BenchmarkScenario:
    """
    A single benchmark scenario combining model, resolution, precision, etc.
    """

    # Identification
    name: str

    # Model configuration
    model: ModelConfig

    # Input configuration
    num_cameras: int = 1  # Number of simultaneous camera angles
    resolution: Tuple[int, int] = (640, 480)  # (width, height)

    # Processing configuration
    precision: str = "fp32"  # "fp16" or "fp32"
    device: str = "mps"  # "mps", "cuda", or "cpu"
    process_res: int = 504  # Internal processing resolution

    # Streaming configuration
    stream_config: StreamConfig = field(default_factory=StreamConfig)

    # Test configuration
    num_frames: int = 100  # Number of frames to process
    warmup_frames: int = 10  # Warmup frames to skip
    sample_every_n_frames: int = 1  # Process every Nth frame (1 = all, 2 = every other, etc.)

    # Pose estimation configuration
    test_pose_estimation: bool = False  # Test camera pose estimation
    test_pose_conditioned: bool = False  # Test pose-conditioned depth
    align_to_input_ext_scale: bool = False  # Align prediction to input pose scale

    # Output configuration
    save_depth_maps: bool = True
    save_comparison_video: bool = True
    save_pose_visualization: bool = False  # Save camera pose visualization

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "name": self.name,
            "model": {
                "name": self.model.name,
                "model_dir": self.model.model_dir,
            },
            "num_cameras": self.num_cameras,
            "resolution": list(self.resolution),
            "precision": self.precision,
            "device": self.device,
            "process_res": self.process_res,
            "stream_config": self.stream_config.to_dict(),
            "num_frames": self.num_frames,
            "warmup_frames": self.warmup_frames,
            "test_pose_estimation": self.test_pose_estimation,
            "test_pose_conditioned": self.test_pose_conditioned,
            "align_to_input_ext_scale": self.align_to_input_ext_scale,
        }


@dataclass
class BenchmarkConfig:
    """
    Complete benchmark configuration defining all scenarios to run.
    """

    # Metadata
    name: str
    description: str = ""

    # Test data
    input_video: Optional[Path] = None  # Path to test video
    input_images: Optional[Path] = None  # Path to test images directory

    # Scenarios to run
    scenarios: List[BenchmarkScenario] = field(default_factory=list)

    # Output configuration
    output_dir: Path = Path("benchmark_results")

    # Comparison settings
    compare_against_baseline: bool = True
    baseline_scenario_name: Optional[str] = None

    # Backend settings
    use_backend: bool = False  # Use backend service for inference
    backend_url: str = "http://localhost:8008"  # Backend service URL
    start_backend: bool = False  # Auto-start backend if not running
    stop_backend: bool = False  # Auto-stop backend after benchmarks

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "input_video": str(self.input_video) if self.input_video else None,
            "input_images": str(self.input_images) if self.input_images else None,
            "output_dir": str(self.output_dir),
            "scenarios": [s.to_dict() for s in self.scenarios],
            "compare_against_baseline": self.compare_against_baseline,
            "baseline_scenario_name": self.baseline_scenario_name,
            "use_backend": self.use_backend,
            "backend_url": self.backend_url,
        }

    @classmethod
    def from_yaml(cls, path: Path) -> "BenchmarkConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Parse scenarios
        scenarios = []
        for scenario_data in data.get("scenarios", []):
            # Parse model
            model_data = scenario_data["model"]
            model = ModelConfig(
                name=model_data["name"],
                model_dir=model_data["model_dir"],
                description=model_data.get("description", ""),
            )

            # Parse stream config
            stream_data = scenario_data.get("stream_config", {})
            stream_config = StreamConfig(**stream_data)

            # Create scenario
            resolution = tuple(scenario_data.get("resolution", [640, 480]))
            # Default process_res to max dimension of resolution (proper scaling)
            default_process_res = max(resolution)

            scenario = BenchmarkScenario(
                name=scenario_data["name"],
                model=model,
                num_cameras=scenario_data.get("num_cameras", 1),
                resolution=resolution,
                precision=scenario_data.get("precision", "fp32"),
                device=scenario_data.get("device", "mps"),
                process_res=scenario_data.get("process_res", default_process_res),
                stream_config=stream_config,
                num_frames=scenario_data.get("num_frames", 100),
                warmup_frames=scenario_data.get("warmup_frames", 10),
                sample_every_n_frames=scenario_data.get("sample_every_n_frames", 1),
                test_pose_estimation=scenario_data.get("test_pose_estimation", False),
                test_pose_conditioned=scenario_data.get("test_pose_conditioned", False),
                align_to_input_ext_scale=scenario_data.get("align_to_input_ext_scale", False),
                save_depth_maps=scenario_data.get("save_depth_maps", True),
                save_comparison_video=scenario_data.get("save_comparison_video", True),
                save_pose_visualization=scenario_data.get("save_pose_visualization", False),
            )
            scenarios.append(scenario)

        # Create config
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            input_video=Path(data["input_video"]) if data.get("input_video") else None,
            input_images=Path(data["input_images"]) if data.get("input_images") else None,
            scenarios=scenarios,
            output_dir=Path(data.get("output_dir", "benchmark_results")),
            compare_against_baseline=data.get("compare_against_baseline", True),
            baseline_scenario_name=data.get("baseline_scenario_name"),
            use_backend=data.get("use_backend", False),
            backend_url=data.get("backend_url", "http://localhost:8008"),
            start_backend=data.get("start_backend", False),
            stop_backend=data.get("stop_backend", False),
        )

    def save_to_yaml(self, path: Path):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


# Predefined model configurations
#
# Model Capabilities Matrix:
# ┌──────────────────┬─────────┬──────────┬───────────┬────────┬───────────┬─────────┐
# │ Model            │ Rel.Dep │ Pose Est │ Pose Cond │ GS     │ Met.Depth │ Sky Seg │
# ├──────────────────┼─────────┼──────────┼───────────┼────────┼───────────┼─────────┤
# │ DA3-SMALL/BASE/  │    ✅   │    ✅    │     ✅    │   ✅*  │           │         │
# │ LARGE (any-view) │         │          │           │        │           │         │
# ├──────────────────┼─────────┼──────────┼───────────┼────────┼───────────┼─────────┤
# │ DA3MONO-LARGE    │    ✅   │          │           │        │           │   ✅    │
# │ (monocular)      │         │          │           │        │           │         │
# ├──────────────────┼─────────┼──────────┼───────────┼────────┼───────────┼─────────┤
# │ DA3METRIC-LARGE  │    ✅   │          │           │        │    ✅     │   ✅    │
# │ (metric)         │         │          │           │        │           │         │
# └──────────────────┴─────────┴──────────┴───────────┴────────┴───────────┴─────────┘
# * GS only on GIANT models

PREDEFINED_MODELS = {
    # Any-view models (support: rel depth, pose est, pose cond)
    "DA3-SMALL": ModelConfig(
        name="DA3-SMALL",
        model_dir="depth-anything/DA3-SMALL",
        description="80M params - Any-view (rel depth + pose est + pose cond)",
    ),
    "DA3-BASE": ModelConfig(
        name="DA3-BASE",
        model_dir="depth-anything/DA3-BASE",
        description="120M params - Any-view (rel depth + pose est + pose cond)",
    ),
    "DA3-LARGE": ModelConfig(
        name="DA3-LARGE",
        model_dir="depth-anything/DA3-LARGE",
        description="350M params - Any-view (rel depth + pose est + pose cond)",
    ),

    # Monocular model (support: rel depth, sky seg - NO pose, NO GS, NO metric)
    "DA3MONO-LARGE": ModelConfig(
        name="DA3MONO-LARGE",
        model_dir="depth-anything/DA3MONO-LARGE",
        description="350M params - Monocular specialist (direct depth + sky seg)",
    ),

    # Metric model (support: metric depth, sky seg - NO pose, NO GS)
    "DA3METRIC-LARGE": ModelConfig(
        name="DA3METRIC-LARGE",
        model_dir="depth-anything/DA3METRIC-LARGE",
        description="350M params - Metric depth specialist (real-world scale + sky seg)",
    ),
}


def create_default_benchmark_config(
    name: str = "Default Streaming Benchmark",
    input_video: Optional[Path] = None,
    device: str = "mps",
) -> BenchmarkConfig:
    """
    Create a default benchmark configuration testing all models.

    Args:
        name: Benchmark name
        input_video: Path to test video
        device: Device to test on

    Returns:
        BenchmarkConfig with predefined scenarios
    """
    scenarios = []

    # Test all models with different configurations
    for model_name, model_config in PREDEFINED_MODELS.items():
        # Single camera, standard resolution (512x512), fp32
        scenarios.append(BenchmarkScenario(
            name=f"{model_name}_1cam_512x512_fp32",
            model=model_config,
            num_cameras=1,
            resolution=(512, 512),
            precision="fp32",
            device=device,
        ))

        # Single camera, fp16 (only on CUDA - MPS doesn't support it well)
        if device == "cuda":
            scenarios.append(BenchmarkScenario(
                name=f"{model_name}_1cam_512x512_fp16",
                model=model_config,
                num_cameras=1,
                resolution=(512, 512),
                precision="fp16",
                device=device,
            ))

        # Multi-camera scenarios disabled - not properly implemented yet
        # TODO: Implement actual multi-camera benchmarking (requires multiple camera angles/views)

    return BenchmarkConfig(
        name=name,
        description=f"Comprehensive streaming benchmark on {device}",
        input_video=input_video,
        scenarios=scenarios,
        baseline_scenario_name="DA3-BASE_1cam_640x480_fp32",
    )
