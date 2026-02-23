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
Unified Inference Service
Provides unified interface for local and remote inference
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import requests
import torch
import typer

from ..api import DepthAnything3
from ..utils.sliding_window import WindowConfig, process_with_sliding_windows


def get_default_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class InferenceService:
    """Unified inference service class"""

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device
        self.model = None

    def load_model(self):
        """Load model"""
        if self.model is None:
            typer.echo(f"Loading model from {self.model_dir}...")
            self.model = DepthAnything3.from_pretrained(self.model_dir).to(self.device)
        return self.model

    def run_local_inference(
        self,
        image_paths: List[str],
        export_dir: str,
        export_format: str = "mini_npz-glb",
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        export_feat_layers: List[int] = None,
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        align_to_input_ext_scale: bool = True,
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        feat_vis_fps: int = 15,
    ) -> Any:
        """Run local inference with automatic batching for large image sets"""
        if export_feat_layers is None:
            export_feat_layers = []

        model = self.load_model()

        #  IMPORTANT: Depth Anything 3 is designed for multi-view 3D reconstruction,
        # not dense video frame processing. The transformer attention has O(N^2) memory.
        # Recommended max views: 2-24 images depending on device.
        if self.device == "mps":
            max_views = 12  # MPS architectural limit
        elif self.device == "cuda":
            max_views = 24  # CUDA can handle more
        else:  # CPU
            max_views = 8

        num_images = len(image_paths)

        # If image count is within limits, process all at once
        if num_images <= max_views:
            # Prepare inference parameters
            inference_kwargs = {
                "image": image_paths,
                "export_dir": export_dir,
                "export_format": export_format,
                "process_res": process_res,
                "process_res_method": process_res_method,
                "export_feat_layers": export_feat_layers,
                "align_to_input_ext_scale": align_to_input_ext_scale,
                "conf_thresh_percentile": conf_thresh_percentile,
                "num_max_points": num_max_points,
                "show_cameras": show_cameras,
                "feat_vis_fps": feat_vis_fps,
                "infer_gs": "gs" in export_format,
            }

            # Add pose data (if exists)
            if extrinsics is not None:
                inference_kwargs["extrinsics"] = extrinsics
            if intrinsics is not None:
                inference_kwargs["intrinsics"] = intrinsics

            # Run inference
            typer.echo(f"Running inference on {num_images} images...")
            prediction = model.inference(**inference_kwargs)

            typer.echo(f"✅ Results saved to {export_dir}")
            typer.echo(f"Export format: {export_format}")

            return prediction
        else:
            # Too many images for the model architecture
            typer.echo("")
            typer.echo(f"❌ ERROR: {num_images} images exceed architectural limit ({max_views} for {self.device})")
            typer.echo("")
            typer.echo(f"🏗️  Depth Anything 3 Architecture:")
            typer.echo(f"   This model uses transformer attention with O(N²) memory complexity.")
            typer.echo(f"   It's designed for multi-view 3D reconstruction (2-{max_views} views),")
            typer.echo(f"   NOT for processing hundreds of dense video frames.")
            typer.echo("")
            typer.echo(f"💡 Solutions for video processing:")
            typer.echo(f"   1. **Reduce FPS drastically**: --fps 0.5 (extracts ~{int(num_images * 0.5 / 10)} frames)")
            typer.echo(f"      For an 18-second video at --fps 1: ~18 frames ✅")
            typer.echo(f"      Current --fps 10: ~{num_images} frames ❌ TOO MANY")
            typer.echo("")
            typer.echo(f"   2. **Extract keyframes only**: Use a separate tool to extract")
            typer.echo(f"      representative frames instead of dense sampling")
            typer.echo("")
            typer.echo(f"   3. **Process short clips**: Split long videos into <{max_views // 2}-second segments")
            typer.echo("")
            typer.echo(f"📊 Current settings:")
            typer.echo(f"   Images: {num_images} | Max for {self.device}: {max_views}")
            typer.echo(f"   Suggested: Reduce --fps to {max(0.1, max_views / (num_images / 10)):.1f} or lower")
            typer.echo("")

            # Exit instead of silently limiting
            raise ValueError(
                f"Too many images ({num_images}) for Depth Anything 3 architecture. "
                f"Maximum {max_views} views supported on {self.device}. "
                f"Reduce --fps parameter."
            )

            # Prepare inference parameters with limited images
            inference_kwargs = {
                "image": image_paths,
                "export_dir": export_dir,
                "export_format": export_format,
                "process_res": process_res,
                "process_res_method": process_res_method,
                "export_feat_layers": export_feat_layers,
                "align_to_input_ext_scale": align_to_input_ext_scale,
                "conf_thresh_percentile": conf_thresh_percentile,
                "num_max_points": num_max_points,
                "show_cameras": show_cameras,
                "feat_vis_fps": feat_vis_fps,
                "infer_gs": "gs" in export_format,
            }

            # Run inference
            typer.echo(f"Running inference on {len(image_paths)} images...")
            prediction = model.inference(**inference_kwargs)

            typer.echo(f"✅ Results saved to {export_dir}")
            typer.echo(f"Export format: {export_format}")

            return prediction

    def run_backend_inference(
        self,
        image_paths: List[str],
        export_dir: str,
        backend_url: str,
        export_format: str = "mini_npz-glb",
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        export_feat_layers: List[int] = None,
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        align_to_input_ext_scale: bool = True,
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        feat_vis_fps: int = 15,
    ) -> Dict[str, Any]:
        """Run backend inference"""
        if export_feat_layers is None:
            export_feat_layers = []

        # Check backend status
        if not self._check_backend_status(backend_url):
            raise typer.BadParameter(f"Backend service is not running at {backend_url}")

        # Prepare payload
        payload = {
            "image_paths": image_paths,
            "export_dir": export_dir,
            "export_format": export_format,
            "process_res": process_res,
            "process_res_method": process_res_method,
            "export_feat_layers": export_feat_layers,
            "align_to_input_ext_scale": align_to_input_ext_scale,
            "conf_thresh_percentile": conf_thresh_percentile,
            "num_max_points": num_max_points,
            "show_cameras": show_cameras,
            "feat_vis_fps": feat_vis_fps,
            "infer_gs": "gs" in export_format,
        }

        # Add pose data (if exists)
        if extrinsics is not None:
            payload["extrinsics"] = [ext.astype(np.float64).tolist() for ext in extrinsics]
        if intrinsics is not None:
            payload["intrinsics"] = [intr.astype(np.float64).tolist() for intr in intrinsics]

        # Submit task
        typer.echo("Submitting inference task to backend...")
        try:
            response = requests.post(f"{backend_url}/inference", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result["success"]:
                task_id = result["task_id"]
                typer.echo("Task submitted successfully!")
                typer.echo(f"Task ID: {task_id}")
                typer.echo(f"Results will be saved to: {export_dir}")
                typer.echo(f"Check backend logs for progress updates with task ID: {task_id}")
                return result
            else:
                raise typer.BadParameter(
                    f"Backend inference submission failed: {result['message']}"
                )
        except requests.exceptions.RequestException as e:
            raise typer.BadParameter(f"Backend inference submission failed: {e}")

    def _check_backend_status(self, backend_url: str) -> bool:
        """Check backend status"""
        try:
            response = requests.get(f"{backend_url}/status", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


def run_inference(
    image_paths: List[str],
    export_dir: str,
    model_dir: str,
    device: Optional[str] = None,
    backend_url: Optional[str] = None,
    export_format: str = "mini_npz-glb",
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    export_feat_layers: List[int] = None,
    extrinsics: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    align_to_input_ext_scale: bool = True,
    conf_thresh_percentile: float = 40.0,
    num_max_points: int = 1_000_000,
    show_cameras: bool = True,
    feat_vis_fps: int = 15,
) -> Union[Any, Dict[str, Any]]:
    """Unified inference interface"""

    # Auto-detect device if not specified
    if device is None:
        device = get_default_device()
        typer.echo(f"🔍 Auto-detected device: {device}")

    service = InferenceService(model_dir, device)

    if backend_url:
        return service.run_backend_inference(
            image_paths=image_paths,
            export_dir=export_dir,
            backend_url=backend_url,
            export_format=export_format,
            process_res=process_res,
            process_res_method=process_res_method,
            export_feat_layers=export_feat_layers,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            align_to_input_ext_scale=align_to_input_ext_scale,
            conf_thresh_percentile=conf_thresh_percentile,
            num_max_points=num_max_points,
            show_cameras=show_cameras,
            feat_vis_fps=feat_vis_fps,
        )
    else:
        return service.run_local_inference(
            image_paths=image_paths,
            export_dir=export_dir,
            export_format=export_format,
            process_res=process_res,
            process_res_method=process_res_method,
            export_feat_layers=export_feat_layers,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            align_to_input_ext_scale=align_to_input_ext_scale,
            conf_thresh_percentile=conf_thresh_percentile,
            num_max_points=num_max_points,
            show_cameras=show_cameras,
            feat_vis_fps=feat_vis_fps,
        )
