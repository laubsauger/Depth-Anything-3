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
Benchmark Runner

Executes benchmark scenarios and collects metrics.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import time
import torch
import numpy as np
import cv2
from tqdm import tqdm

from ..api import DepthAnything3
from ..services.streaming import StreamingDepthEstimator, StreamConfig as ServicesStreamConfig
from .config import BenchmarkScenario, BenchmarkConfig
from .metrics import MetricsCollector, BenchmarkMetrics
from .backend_client import BackendClient
from .backend_manager import BackendManager


class BenchmarkRunner:
    """
    Runs benchmarks and collects metrics.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkMetrics] = []
        self.backend_client: Optional[BackendClient] = None
        self.backend_manager: Optional[BackendManager] = None

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize backend if needed
        if self.config.use_backend:
            print(f"🔌 Using backend at {self.config.backend_url}")

            # Auto-start backend if requested
            if self.config.start_backend:
                # Get first scenario's model for backend
                if not self.config.scenarios:
                    raise ValueError("No scenarios defined, cannot start backend")

                first_scenario = self.config.scenarios[0]
                self.backend_manager = BackendManager(
                    model_dir=first_scenario.model.model_dir,
                    backend_url=self.config.backend_url,
                    device=first_scenario.device,
                )

                if not self.backend_manager.start():
                    raise RuntimeError("Failed to start backend service")

            # Initialize client
            self.backend_client = BackendClient(self.config.backend_url)

            # Health check
            if not self.backend_client.health_check():
                raise RuntimeError(
                    f"Backend at {self.config.backend_url} is not responding. "
                    "Please start the backend service first: "
                    f"da3 backend --model-dir <model_dir>"
                )

            # Show backend status
            status = self.backend_client.get_status()
            print(f"✅ Backend ready: {status.get('model', 'unknown')} on {status.get('device', 'unknown')}")

    def _infer(
        self,
        model: Optional[DepthAnything3],
        frames: List[np.ndarray],
        process_res: int,
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        align_to_input_ext_scale: bool = False,
    ):
        """
        Run inference using either local model or backend.

        Args:
            model: Local model (None if using backend)
            frames: Input frames
            process_res: Processing resolution
            extrinsics: Camera extrinsics (optional)
            intrinsics: Camera intrinsics (optional)
            align_to_input_ext_scale: Align to input pose scale

        Returns:
            Prediction object
        """
        if self.backend_client:
            # Use backend
            return self.backend_client.inference(
                images=frames,
                process_res=process_res,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                align_to_input_ext_scale=align_to_input_ext_scale,
            )
        else:
            # Use local model
            return model.inference(
                image=frames,
                export_dir=None,
                process_res=process_res,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                align_to_input_ext_scale=align_to_input_ext_scale,
            )

    def run_all(self) -> List[BenchmarkMetrics]:
        """Run all benchmark scenarios."""
        print(f"🚀 Starting benchmark: {self.config.name}")
        print(f"📁 Output directory: {self.config.output_dir}")
        print(f"📊 Running {len(self.config.scenarios)} scenarios\n")

        try:
            for scenario in self.config.scenarios:
                print(f"\n{'=' * 70}")
                print(f"Running scenario: {scenario.name}")
                print(f"{'=' * 70}")

                try:
                    metrics = self.run_scenario(scenario)
                    self.results.append(metrics)

                    # Print summary
                    print(f"\n✅ Scenario complete!")
                    print(metrics)

                except Exception as e:
                    print(f"\n❌ Scenario failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        finally:
            # Cleanup backend if we started it
            if self.config.stop_backend and self.backend_manager:
                self.backend_manager.stop()

        return self.results

    def run_scenario(self, scenario: BenchmarkScenario) -> BenchmarkMetrics:
        """Run a single benchmark scenario."""
        # Create scenario output directory
        scenario_dir = self.config.output_dir / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics collector
        collector = MetricsCollector(scenario.name, scenario.device)

        # Load model (or use backend)
        model = None
        if self.backend_client:
            print(f"🔌 Using backend service (model reuse enabled)")
        else:
            print(f"📦 Loading model: {scenario.model.name}")
            model = DepthAnything3.from_pretrained(scenario.model.model_dir)
            model = model.to(scenario.device)

            # Set precision (fp16 only supported on CUDA)
            if scenario.precision == "fp16":
                if scenario.device == "cuda":
                    model = model.half()
                else:
                    print(f"⚠️  fp16 not supported on {scenario.device}, using fp32")
                    scenario.precision = "fp32"  # Update for metrics

        # Load test frames
        frames = self._load_test_frames(
            scenario.num_frames,
            scenario.resolution,
            scenario.num_cameras,
        )

        # Sample frames if configured
        if scenario.sample_every_n_frames > 1:
            frames = frames[::scenario.sample_every_n_frames]
            print(f"📹 Loaded {len(frames)} test frames (sampled every {scenario.sample_every_n_frames} frames)")
        else:
            print(f"📹 Loaded {len(frames)} test frames")

        print(f"🎯 Input Resolution: {frames[0].shape[1]}x{frames[0].shape[0]} (upper bound: {scenario.resolution})")
        print(f"⚙️  Model Process Resolution: {scenario.process_res}")
        print(f"🔧 Device: {scenario.device}")
        print(f"📷 Cameras: {scenario.num_cameras}")
        print(f"🔢 Precision: {scenario.precision}")
        if scenario.sample_every_n_frames > 1:
            print(f"⏭️  Frame Sampling: Every {scenario.sample_every_n_frames} frames")
        print(f"🪟 Window Size: {scenario.stream_config.window_size}")
        print(f"🔄 Overlap: {scenario.stream_config.overlap}")

        # Check if pose estimation is enabled
        if scenario.test_pose_estimation or scenario.test_pose_conditioned:
            print(f"🎯 Pose Mode: {'Conditioned' if scenario.test_pose_conditioned else 'Estimation Only'}")
            depth_maps, estimated_poses = self._run_pose_benchmark(
                model, scenario, frames, collector
            )
            total_time = collector.metrics.total_time_s
            metrics = collector.metrics

            # Save pose visualization if requested
            if scenario.save_pose_visualization and estimated_poses is not None:
                self._save_pose_visualization(estimated_poses, scenario_dir)

        else:
            # Standard streaming mode (or batch if using backend)
            if self.backend_client:
                # Backend doesn't support streaming, use batch inference
                print(f"📦 Mode: Batch inference via backend")

                # Warmup
                print(f"\n🔥 Warming up...")
                warmup_frames = frames[:min(scenario.warmup_frames, len(frames))]
                _ = self._infer(model, warmup_frames, scenario.process_res)

                # Benchmark
                print(f"\n🏃 Running benchmark...")
                start_time = time.time()

                prediction = self._infer(model, frames, scenario.process_res)

                total_time = time.time() - start_time

                depth_maps = [prediction.depth[i] for i in range(len(frames))]

                # Finalize metrics
                metrics = collector.finalize(total_time)

            else:
                # Local model with streaming
                print(f"🌊 Mode: Streaming (no pose)")
                stream_config = ServicesStreamConfig(
                    buffer_size=scenario.stream_config.buffer_size or 12,
                    overlap=scenario.stream_config.overlap or 4,
                    output_latency_frames=scenario.stream_config.output_latency_frames,
                )
                estimator = StreamingDepthEstimator(
                    model=model,
                    device=scenario.device,
                    config=stream_config,
                    process_res=scenario.process_res,
                    collect_timing=True,  # Always collect timing in benchmarks
                )

                # Warmup
                print(f"\n🔥 Warming up with {scenario.warmup_frames} frames...")
                for i in range(min(scenario.warmup_frames, len(frames))):
                    estimator.process_frame(frames[i])

                # Reset estimator after warmup
                estimator.reset()

                # Benchmark
                print(f"\n🏃 Running benchmark...")
                start_time = time.time()

                depth_maps = []

                for frame_idx, frame in enumerate(tqdm(frames, desc="Processing frames")):
                    collector.start_frame(frame_idx)

                    # Process frame
                    collector.start_inference()
                    depth = estimator.process_frame(frame)
                    collector.end_inference()

                    collector.end_frame()

                    if depth is not None:
                        depth_maps.append(depth)

                # Flush remaining frames
                for depth in estimator.flush():
                    depth_maps.append(depth)

                total_time = time.time() - start_time

                # Finalize metrics
                metrics = collector.finalize(total_time)

                # Transfer timing stats from estimator to metrics
                if estimator.timing_stats is not None:
                    metrics.timing_stats = estimator.timing_stats

                del estimator

        # Calculate temporal consistency metrics
        self._calculate_temporal_metrics(depth_maps, metrics)

        # Save depth maps if requested
        if scenario.save_depth_maps:
            self._save_depth_maps(depth_maps, scenario_dir)

        # Save comparison video if requested
        if scenario.save_comparison_video:
            self._save_comparison_video(frames, depth_maps, scenario_dir)

        # Save temporal analysis
        self._save_temporal_analysis(depth_maps, scenario_dir)

        # Save metrics
        self._save_metrics(metrics, scenario_dir)

        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return metrics

    def _run_pose_benchmark(
        self,
        model: DepthAnything3,
        scenario: BenchmarkScenario,
        frames: List[np.ndarray],
        collector: MetricsCollector,
    ) -> tuple[List[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """
        Run benchmark with pose estimation.

        Returns:
            depth_maps: List of depth predictions
            estimated_poses: Dict with 'extrinsics' and 'intrinsics' arrays
        """
        print(f"\n🔥 Warming up...")
        # Warmup with a subset
        warmup_frames = frames[:min(scenario.warmup_frames, len(frames))]
        _ = self._infer(model, warmup_frames, scenario.process_res)

        print(f"\n🏃 Running pose estimation benchmark...")
        start_time = time.time()

        estimated_poses = None

        if scenario.test_pose_conditioned:
            # Two-pass: First estimate poses, then use them for depth
            print("  Pass 1/2: Estimating camera poses...")

            pass1_start = time.time()
            pose_prediction = self._infer(model, frames, scenario.process_res)
            pass1_time = time.time() - pass1_start

            # Extract estimated poses
            estimated_extrinsics = pose_prediction.extrinsics  # (N, 4, 4)
            estimated_intrinsics = pose_prediction.intrinsics  # (N, 3, 3)

            estimated_poses = {
                'extrinsics': estimated_extrinsics,
                'intrinsics': estimated_intrinsics,
            }

            print(f"  ✅ Poses estimated in {pass1_time:.2f}s")
            print(f"  Pass 2/2: Pose-conditioned depth estimation...")

            pass2_start = time.time()
            final_prediction = self._infer(
                model,
                frames,
                scenario.process_res,
                extrinsics=estimated_extrinsics,
                intrinsics=estimated_intrinsics,
                align_to_input_ext_scale=scenario.align_to_input_ext_scale,
            )
            pass2_time = time.time() - pass2_start

            print(f"  ✅ Depth estimated in {pass2_time:.2f}s")

            depth_maps = [final_prediction.depth[i] for i in range(len(frames))]

            # Update metrics
            collector.metrics.pose_estimation_enabled = True
            collector.metrics.pose_conditioned_enabled = True

        else:
            # Single pass: Just estimate poses (and get depth as byproduct)
            print("  Estimating poses and depth...")

            prediction = self._infer(model, frames, scenario.process_res)

            depth_maps = [prediction.depth[i] for i in range(len(frames))]

            estimated_poses = {
                'extrinsics': prediction.extrinsics,
                'intrinsics': prediction.intrinsics,
            }

            # Update metrics
            collector.metrics.pose_estimation_enabled = True
            collector.metrics.pose_conditioned_enabled = False

        total_time = time.time() - start_time
        collector.metrics.total_time_s = total_time
        collector.metrics.total_frames = len(frames)

        # Compute FPS
        collector.metrics.avg_fps = len(frames) / total_time if total_time > 0 else 0.0
        collector.metrics.min_fps = collector.metrics.avg_fps
        collector.metrics.max_fps = collector.metrics.avg_fps

        print(f"\n✅ Pose benchmark complete: {collector.metrics.avg_fps:.2f} FPS")

        return depth_maps, estimated_poses

    def _load_test_frames(
        self,
        num_frames: int,
        resolution: tuple,
        num_cameras: int,
    ) -> List[np.ndarray]:
        """Load test frames from input video or images."""
        frames = []

        if self.config.input_video:
            # Load from video
            cap = cv2.VideoCapture(str(self.config.input_video))

            frame_count = 0
            while len(frames) < num_frames:
                ret, frame = cap.read()
                if not ret:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize preserving aspect ratio (resolution is upper bound)
                h, w = frame.shape[:2]
                target_w, target_h = resolution
                scale = min(target_w / w, target_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                frames.append(frame)
                frame_count += 1

            cap.release()

        elif self.config.input_images:
            # Load from images directory
            image_files = sorted(self.config.input_images.glob("*"))
            image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]

            for i in range(num_frames):
                img_path = image_files[i % len(image_files)]
                frame = cv2.imread(str(img_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize preserving aspect ratio (resolution is upper bound)
                h, w = frame.shape[:2]
                target_w, target_h = resolution
                scale = min(target_w / w, target_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                frames.append(frame)

        else:
            # Generate synthetic frames
            print("⚠️  No input provided, generating synthetic frames")
            for i in range(num_frames):
                # Create a gradient pattern
                frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
                frame[:, :, 0] = np.linspace(0, 255, resolution[0], dtype=np.uint8)
                frame[:, :, 1] = np.linspace(0, 255, resolution[1], dtype=np.uint8).reshape(-1, 1)
                frames.append(frame)

        return frames

    def _save_depth_maps(self, depth_maps: List[np.ndarray], output_dir: Path):
        """Save depth maps as images."""
        depth_dir = output_dir / "depth_maps"
        depth_dir.mkdir(exist_ok=True)

        for i, depth in enumerate(depth_maps):
            # Normalize to 0-255
            depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)

            # Save as PNG
            cv2.imwrite(str(depth_dir / f"depth_{i:04d}.png"), depth_norm)

        print(f"💾 Saved {len(depth_maps)} depth maps to {depth_dir}")

    def _save_comparison_video(
        self,
        input_frames: List[np.ndarray],
        depth_maps: List[np.ndarray],
        output_dir: Path,
    ):
        """Create side-by-side comparison video."""
        if not depth_maps:
            return

        output_path = output_dir / "comparison.mp4"

        # Get dimensions
        h, w = input_frames[0].shape[:2]

        # Create video writer with H.264 codec (more compatible)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
        out = cv2.VideoWriter(str(output_path), fourcc, 15.0, (w * 2, h))

        # Ensure same number of frames
        num_frames = min(len(input_frames), len(depth_maps))

        for i in range(num_frames):
            input_frame = input_frames[i]
            depth_map = depth_maps[i]

            # Normalize depth to 0-255
            depth_norm = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)

            # Resize depth to match input frame dimensions
            depth_resized = cv2.resize(depth_norm, (w, h), interpolation=cv2.INTER_LINEAR)

            # Apply heatmap colormap (TURBO is perceptually uniform and vibrant)
            depth_heatmap = cv2.applyColorMap(depth_resized, cv2.COLORMAP_TURBO)

            # Convert input to BGR
            input_bgr = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR)

            # Create side-by-side: [Input | Depth Heatmap]
            combined = np.hstack([input_bgr, depth_heatmap])

            out.write(combined)

        out.release()
        print(f"🎥 Saved comparison video to {output_path}")

    def _save_metrics(self, metrics: BenchmarkMetrics, output_dir: Path):
        """Save metrics to JSON file."""
        import json

        output_path = output_dir / "metrics.json"

        with open(output_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)

        print(f"📊 Saved metrics to {output_path}")

    def _save_pose_visualization(
        self,
        estimated_poses: Dict[str, np.ndarray],
        output_dir: Path,
    ):
        """
        Save 3D visualization of estimated camera trajectory.

        Args:
            estimated_poses: Dict with 'extrinsics' (N, 4, 4) and 'intrinsics' (N, 3, 3)
            output_dir: Output directory
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("⚠️  Matplotlib not available, skipping pose visualization")
            return

        extrinsics = estimated_poses['extrinsics']  # (N, 4, 4)

        # Extract camera positions (translation vectors)
        # Extrinsics are world-to-camera, so camera position in world = -R^T @ t
        positions = []
        for i in range(len(extrinsics)):
            R = extrinsics[i, :3, :3]
            t = extrinsics[i, :3, 3]
            cam_pos = -R.T @ t
            positions.append(cam_pos)

        positions = np.array(positions)  # (N, 3)

        # Create 3D trajectory plot
        fig = plt.figure(figsize=(15, 10))

        # 3D view
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                 'b-', linewidth=2, label='Camera Path', alpha=0.6)
        ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                    c=range(len(positions)), cmap='viridis', s=50, alpha=0.8)
        ax1.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]],
                    c='green', s=200, marker='*', label='Start', zorder=10)
        ax1.scatter([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]],
                    c='red', s=200, marker='X', label='End', zorder=10)

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Camera Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Top-down view (XY plane)
        ax2 = fig.add_subplot(222)
        ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.6)
        scatter = ax2.scatter(positions[:, 0], positions[:, 1],
                              c=range(len(positions)), cmap='viridis', s=30)
        ax2.scatter(positions[0, 0], positions[0, 1],
                    c='green', s=150, marker='*', label='Start', zorder=10)
        ax2.scatter(positions[-1, 0], positions[-1, 1],
                    c='red', s=150, marker='X', label='End', zorder=10)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top-Down View (XY)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')

        # Side view (XZ plane)
        ax3 = fig.add_subplot(223)
        ax3.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2, alpha=0.6)
        ax3.scatter(positions[:, 0], positions[:, 2],
                    c=range(len(positions)), cmap='viridis', s=30)
        ax3.scatter(positions[0, 0], positions[0, 2],
                    c='green', s=150, marker='*', label='Start', zorder=10)
        ax3.scatter(positions[-1, 0], positions[-1, 2],
                    c='red', s=150, marker='X', label='End', zorder=10)
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('Side View (XZ)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Front view (YZ plane)
        ax4 = fig.add_subplot(224)
        ax4.plot(positions[:, 1], positions[:, 2], 'b-', linewidth=2, alpha=0.6)
        ax4.scatter(positions[:, 1], positions[:, 2],
                    c=range(len(positions)), cmap='viridis', s=30)
        ax4.scatter(positions[0, 1], positions[0, 2],
                    c='green', s=150, marker='*', label='Start', zorder=10)
        ax4.scatter(positions[-1, 1], positions[-1, 2],
                    c='red', s=150, marker='X', label='End', zorder=10)
        ax4.set_xlabel('Y (m)')
        ax4.set_ylabel('Z (m)')
        ax4.set_title('Front View (YZ)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_path = output_dir / "camera_trajectory.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📸 Saved camera trajectory visualization to {output_path}")

        # Save pose statistics
        self._save_pose_stats(positions, extrinsics, output_dir)

    def _save_pose_stats(
        self,
        positions: np.ndarray,
        extrinsics: np.ndarray,
        output_dir: Path,
    ):
        """Save pose statistics to text file."""
        # Compute statistics
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_distance = np.sum(distances)
        avg_distance = np.mean(distances)

        # Compute rotation changes
        rotations = []
        for i in range(len(extrinsics) - 1):
            R1 = extrinsics[i, :3, :3]
            R2 = extrinsics[i + 1, :3, :3]
            # Relative rotation
            R_rel = R2 @ R1.T
            # Rotation angle in degrees
            trace = np.trace(R_rel)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            rotations.append(np.degrees(angle))

        avg_rotation = np.mean(rotations) if rotations else 0.0

        # Write statistics
        stats_path = output_dir / "pose_statistics.txt"
        with open(stats_path, 'w') as f:
            f.write("Camera Pose Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Frames: {len(positions)}\n")
            f.write(f"Total Distance Traveled: {total_distance:.3f} m\n")
            f.write(f"Average Distance per Frame: {avg_distance:.3f} m\n")
            f.write(f"Average Rotation per Frame: {avg_rotation:.2f}°\n\n")

            f.write("Trajectory Bounds:\n")
            f.write(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}] m\n")
            f.write(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}] m\n")
            f.write(f"  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}] m\n")

        print(f"📊 Saved pose statistics to {stats_path}")

    def _calculate_temporal_metrics(
        self, depth_maps: List[np.ndarray], metrics: BenchmarkMetrics
    ):
        """Calculate temporal consistency metrics from depth sequence."""
        if len(depth_maps) < 2:
            return

        # Calculate frame-to-frame differences
        diffs = []
        for i in range(len(depth_maps) - 1):
            d1 = depth_maps[i]
            d2 = depth_maps[i + 1]
            diff = np.abs(d2 - d1)
            diffs.append(np.mean(diff))

        diffs = np.array(diffs)

        # Temporal jitter: std dev of changes
        metrics.temporal_jitter = float(np.std(diffs))

        # Temporal smoothness: 1 - normalized mean diff
        metrics.temporal_smoothness = float(1.0 - (np.mean(diffs) / (np.mean([d.max() - d.min() for d in depth_maps]) + 1e-6)))

        # Flicker score: high-frequency changes
        # Compare every other frame to detect rapid oscillations
        if len(depth_maps) >= 3:
            flickers = []
            for i in range(0, len(depth_maps) - 2, 2):
                d1 = depth_maps[i]
                d2 = depth_maps[i + 1]
                d3 = depth_maps[i + 2]
                # Detect back-and-forth changes
                diff_12 = np.mean(np.abs(d2 - d1))
                diff_23 = np.mean(np.abs(d3 - d2))
                flickers.append((diff_12 + diff_23) / 2)
            metrics.flicker_score = float(np.mean(flickers)) if flickers else 0.0

    def _save_temporal_analysis(self, depth_maps: List[np.ndarray], output_dir: Path):
        """Save temporal consistency analysis with visualizations."""
        if len(depth_maps) < 2:
            return

        import matplotlib.pyplot as plt

        # Calculate frame-to-frame changes
        changes = []
        for i in range(len(depth_maps) - 1):
            diff = np.mean(np.abs(depth_maps[i + 1] - depth_maps[i]))
            changes.append(diff)

        # Create analysis plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Frame-to-frame changes
        ax1.plot(changes, 'b-', alpha=0.7, linewidth=1)
        ax1.axhline(y=np.mean(changes), color='r', linestyle='--', label=f'Mean: {np.mean(changes):.4f}')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Mean Absolute Difference')
        ax1.set_title('Temporal Consistency: Frame-to-Frame Depth Changes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Histogram of changes
        ax2.hist(changes, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.mean(changes), color='r', linestyle='--', label=f'Mean: {np.mean(changes):.4f}')
        ax2.axvline(x=np.median(changes), color='g', linestyle='--', label=f'Median: {np.median(changes):.4f}')
        ax2.set_xlabel('Mean Absolute Difference')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Depth Changes')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = output_dir / "temporal_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Save sample frames for visual comparison
        sample_indices = [0, len(depth_maps) // 4, len(depth_maps) // 2, 3 * len(depth_maps) // 4, -1]
        fig, axes = plt.subplots(1, len(sample_indices), figsize=(15, 3))
        for idx, frame_idx in enumerate(sample_indices):
            if frame_idx >= len(depth_maps):
                continue
            depth = depth_maps[frame_idx]
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            axes[idx].imshow(depth_norm, cmap='turbo')
            axes[idx].set_title(f'Frame {frame_idx if frame_idx >= 0 else len(depth_maps) + frame_idx}')
            axes[idx].axis('off')

        plt.tight_layout()
        sample_path = output_dir / "temporal_samples.png"
        plt.savefig(sample_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📊 Saved temporal analysis to {output_path}")
        print(f"📊 Saved temporal samples to {sample_path}")
