# flake8: noqa: E402
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
Refactored Depth Anything 3 CLI
Clean, modular command-line interface
"""

from __future__ import annotations

import os
import torch
import typer

from depth_anything_3.services import start_server
from depth_anything_3.services.gallery import gallery as gallery_main
from depth_anything_3.services.inference_service import run_inference
from depth_anything_3.services.stream_server import start_stream_server
from depth_anything_3.services.input_handlers import (
    ColmapHandler,
    ImageHandler,
    ImagesHandler,
    InputHandler,
    VideoHandler,
    parse_export_feat,
)
from depth_anything_3.utils.constants import (
    DEFAULT_EXPORT_DIR,
    DEFAULT_GALLERY_DIR,
    DEFAULT_GRADIO_DIR,
    DEFAULT_MODEL,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

app = typer.Typer(help="Depth Anything 3 - Video depth estimation CLI", add_completion=False)


def get_default_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# ============================================================================
# Input type detection utilities
# ============================================================================

# Supported file extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}


def detect_input_type(input_path: str) -> str:
    """
    Detect input type from path.

    Returns:
        - "image": Single image file
        - "images": Directory containing images
        - "video": Video file
        - "colmap": COLMAP directory structure
        - "unknown": Cannot determine type
    """
    if not os.path.exists(input_path):
        return "unknown"

    # Check if it's a file
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            return "image"
        elif ext in VIDEO_EXTENSIONS:
            return "video"
        return "unknown"

    # Check if it's a directory
    if os.path.isdir(input_path):
        # Check for COLMAP structure
        images_dir = os.path.join(input_path, "images")
        sparse_dir = os.path.join(input_path, "sparse")

        if os.path.isdir(images_dir) and os.path.isdir(sparse_dir):
            return "colmap"

        # Check if directory contains image files
        for item in os.listdir(input_path):
            item_path = os.path.join(input_path, item)
            if os.path.isfile(item_path):
                ext = os.path.splitext(item)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    return "images"

        return "unknown"

    return "unknown"


# ============================================================================
# Common parameters and configuration
# ============================================================================

# ============================================================================
# Inference commands
# ============================================================================


@app.command()
def auto(
    input_path: str = typer.Argument(
        ..., help="Path to input (image, directory, video, or COLMAP)"
    ),
    model_dir: str = typer.Option(DEFAULT_MODEL, help="Model directory path"),
    export_dir: str = typer.Option(DEFAULT_EXPORT_DIR, help="Export directory"),
    export_format: str = typer.Option("glb", help="Export format"),
    device: str = typer.Option(None, help="Device to use (auto-detects if not specified)"),
    use_backend: bool = typer.Option(False, help="Use backend service for inference"),
    backend_url: str = typer.Option(
        "http://localhost:8008", help="Backend URL (default: http://localhost:8008)"
    ),
    process_res: int = typer.Option(504, help="Processing resolution"),
    process_res_method: str = typer.Option(
        "upper_bound_resize", help="Processing resolution method"
    ),
    export_feat: str = typer.Option(
        "",
        help="[FEAT_VIS]Export features from specified layers using comma-separated indices (e.g., '0,1,2').",
    ),
    auto_cleanup: bool = typer.Option(
        False, help="Automatically clean export directory if it exists (no prompt)"
    ),
    # Video-specific options
    fps: float = typer.Option(1.0, help="[Video] Sampling FPS for frame extraction"),
    # COLMAP-specific options
    sparse_subdir: str = typer.Option(
        "", help="[COLMAP] Sparse reconstruction subdirectory (e.g., '0' for sparse/0/)"
    ),
    align_to_input_ext_scale: bool = typer.Option(
        True, help="[COLMAP] Align prediction to input extrinsics scale"
    ),
    # GLB export options
    conf_thresh_percentile: float = typer.Option(
        40.0, help="[GLB] Lower percentile for adaptive confidence threshold"
    ),
    num_max_points: int = typer.Option(
        1_000_000, help="[GLB] Maximum number of points in the point cloud"
    ),
    show_cameras: bool = typer.Option(
        True, help="[GLB] Show camera wireframes in the exported scene"
    ),
    # Feat_vis export options
    feat_vis_fps: int = typer.Option(15, help="[FEAT_VIS] Frame rate for output video"),
):
    """
    Automatically detect input type and run appropriate processing.

    Supports:
    - Single image file (.jpg, .png, etc.)
    - Directory of images
    - Video file (.mp4, .avi, etc.)
    - COLMAP directory (with 'images' and 'sparse' subdirectories)
    """
    # Detect input type
    input_type = detect_input_type(input_path)

    if input_type == "unknown":
        typer.echo(f"❌ Error: Cannot determine input type for: {input_path}", err=True)
        typer.echo("Supported inputs:", err=True)
        typer.echo("  - Single image file (.jpg, .png, etc.)", err=True)
        typer.echo("  - Directory containing images", err=True)
        typer.echo("  - Video file (.mp4, .avi, etc.)", err=True)
        typer.echo("  - COLMAP directory (with 'images/' and 'sparse/' subdirectories)", err=True)
        raise typer.Exit(1)

    # Display detected type
    typer.echo(f"🔍 Detected input type: {input_type.upper()}")
    typer.echo(f"📁 Input path: {input_path}")
    typer.echo()

    # Determine backend URL based on use_backend flag
    final_backend_url = backend_url if use_backend else None

    # Parse export_feat parameter
    export_feat_layers = parse_export_feat(export_feat)

    # Route to appropriate handler
    if input_type == "image":
        typer.echo("Processing single image...")
        # Process input
        image_files = ImageHandler.process(input_path)

        # Handle export directory
        export_dir = InputHandler.handle_export_dir(export_dir, auto_cleanup)

        # Run inference
        run_inference(
            image_paths=image_files,
            export_dir=export_dir,
            model_dir=model_dir,
            device=device,
            backend_url=final_backend_url,
            export_format=export_format,
            process_res=process_res,
            process_res_method=process_res_method,
            export_feat_layers=export_feat_layers,
            conf_thresh_percentile=conf_thresh_percentile,
            num_max_points=num_max_points,
            show_cameras=show_cameras,
            feat_vis_fps=feat_vis_fps,
        )

    elif input_type == "images":
        typer.echo("Processing directory of images...")
        # Process input - use default extensions
        image_files = ImagesHandler.process(input_path, "png,jpg,jpeg")

        # Handle export directory
        export_dir = InputHandler.handle_export_dir(export_dir, auto_cleanup)

        # Run inference
        run_inference(
            image_paths=image_files,
            export_dir=export_dir,
            model_dir=model_dir,
            device=device,
            backend_url=final_backend_url,
            export_format=export_format,
            process_res=process_res,
            process_res_method=process_res_method,
            export_feat_layers=export_feat_layers,
            conf_thresh_percentile=conf_thresh_percentile,
            num_max_points=num_max_points,
            show_cameras=show_cameras,
            feat_vis_fps=feat_vis_fps,
        )

    elif input_type == "video":
        typer.echo(f"Processing video with FPS={fps}...")
        # Handle export directory
        export_dir = InputHandler.handle_export_dir(export_dir, auto_cleanup)

        # Process input
        image_files = VideoHandler.process(input_path, export_dir, fps)

        # Run inference
        run_inference(
            image_paths=image_files,
            export_dir=export_dir,
            model_dir=model_dir,
            device=device,
            backend_url=final_backend_url,
            export_format=export_format,
            process_res=process_res,
            process_res_method=process_res_method,
            export_feat_layers=export_feat_layers,
            conf_thresh_percentile=conf_thresh_percentile,
            num_max_points=num_max_points,
            show_cameras=show_cameras,
            feat_vis_fps=feat_vis_fps,
        )

    elif input_type == "colmap":
        typer.echo(
            f"Processing COLMAP directory (sparse subdirectory: '{sparse_subdir or 'default'}')..."
        )
        # Process input
        image_files, extrinsics, intrinsics = ColmapHandler.process(input_path, sparse_subdir)

        # Handle export directory
        export_dir = InputHandler.handle_export_dir(export_dir, auto_cleanup)

        # Run inference
        run_inference(
            image_paths=image_files,
            export_dir=export_dir,
            model_dir=model_dir,
            device=device,
            backend_url=final_backend_url,
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

    typer.echo()
    typer.echo("✅ Processing completed successfully!")


@app.command()
def image(
    image_path: str = typer.Argument(..., help="Path to input image file"),
    model_dir: str = typer.Option(DEFAULT_MODEL, help="Model directory path"),
    export_dir: str = typer.Option(DEFAULT_EXPORT_DIR, help="Export directory"),
    export_format: str = typer.Option("glb", help="Export format"),
    device: str = typer.Option(None, help="Device to use (auto-detects if not specified)"),
    use_backend: bool = typer.Option(False, help="Use backend service for inference"),
    backend_url: str = typer.Option(
        "http://localhost:8008", help="Backend URL (default: http://localhost:8008)"
    ),
    process_res: int = typer.Option(504, help="Processing resolution"),
    process_res_method: str = typer.Option(
        "upper_bound_resize", help="Processing resolution method"
    ),
    export_feat: str = typer.Option(
        "",
        help="[FEAT_VIS] Export features from specified layers using comma-separated indices (e.g., '0,1,2').",
    ),
    auto_cleanup: bool = typer.Option(
        False, help="Automatically clean export directory if it exists (no prompt)"
    ),
    # GLB export options
    conf_thresh_percentile: float = typer.Option(
        40.0, help="[GLB] Lower percentile for adaptive confidence threshold"
    ),
    num_max_points: int = typer.Option(
        1_000_000, help="[GLB] Maximum number of points in the point cloud"
    ),
    show_cameras: bool = typer.Option(
        True, help="[GLB] Show camera wireframes in the exported scene"
    ),
    # Feat_vis export options
    feat_vis_fps: int = typer.Option(15, help="[FEAT_VIS] Frame rate for output video"),
):
    """Run camera pose and depth estimation on a single image."""
    # Process input
    image_files = ImageHandler.process(image_path)

    # Handle export directory
    export_dir = InputHandler.handle_export_dir(export_dir, auto_cleanup)

    # Parse export_feat parameter
    export_feat_layers = parse_export_feat(export_feat)

    # Determine backend URL based on use_backend flag
    final_backend_url = backend_url if use_backend else None

    # Run inference
    run_inference(
        image_paths=image_files,
        export_dir=export_dir,
        model_dir=model_dir,
        device=device,
        backend_url=final_backend_url,
        export_format=export_format,
        process_res=process_res,
        process_res_method=process_res_method,
        export_feat_layers=export_feat_layers,
        conf_thresh_percentile=conf_thresh_percentile,
        num_max_points=num_max_points,
        show_cameras=show_cameras,
        feat_vis_fps=feat_vis_fps,
    )


@app.command()
def images(
    images_dir: str = typer.Argument(..., help="Path to directory containing input images"),
    image_extensions: str = typer.Option(
        "png,jpg,jpeg", help="Comma-separated image file extensions to process"
    ),
    model_dir: str = typer.Option(DEFAULT_MODEL, help="Model directory path"),
    export_dir: str = typer.Option(DEFAULT_EXPORT_DIR, help="Export directory"),
    export_format: str = typer.Option("glb", help="Export format"),
    device: str = typer.Option(None, help="Device to use (auto-detects if not specified)"),
    use_backend: bool = typer.Option(False, help="Use backend service for inference"),
    backend_url: str = typer.Option(
        "http://localhost:8008", help="Backend URL (default: http://localhost:8008)"
    ),
    process_res: int = typer.Option(504, help="Processing resolution"),
    process_res_method: str = typer.Option(
        "upper_bound_resize", help="Processing resolution method"
    ),
    export_feat: str = typer.Option(
        "",
        help="[FEAT_VIS] Export features from specified layers using comma-separated indices (e.g., '0,1,2').",
    ),
    auto_cleanup: bool = typer.Option(
        False, help="Automatically clean export directory if it exists (no prompt)"
    ),
    # GLB export options
    conf_thresh_percentile: float = typer.Option(
        40.0, help="[GLB] Lower percentile for adaptive confidence threshold"
    ),
    num_max_points: int = typer.Option(
        1_000_000, help="[GLB] Maximum number of points in the point cloud"
    ),
    show_cameras: bool = typer.Option(
        True, help="[GLB] Show camera wireframes in the exported scene"
    ),
    # Feat_vis export options
    feat_vis_fps: int = typer.Option(15, help="[FEAT_VIS] Frame rate for output video"),
):
    """Run camera pose and depth estimation on a directory of images."""
    # Process input
    image_files = ImagesHandler.process(images_dir, image_extensions)

    # Handle export directory
    export_dir = InputHandler.handle_export_dir(export_dir, auto_cleanup)

    # Parse export_feat parameter
    export_feat_layers = parse_export_feat(export_feat)

    # Determine backend URL based on use_backend flag
    final_backend_url = backend_url if use_backend else None

    # Run inference
    run_inference(
        image_paths=image_files,
        export_dir=export_dir,
        model_dir=model_dir,
        device=device,
        backend_url=final_backend_url,
        export_format=export_format,
        process_res=process_res,
        process_res_method=process_res_method,
        export_feat_layers=export_feat_layers,
        conf_thresh_percentile=conf_thresh_percentile,
        num_max_points=num_max_points,
        show_cameras=show_cameras,
        feat_vis_fps=feat_vis_fps,
    )


@app.command()
def colmap(
    colmap_dir: str = typer.Argument(
        ..., help="Path to COLMAP directory containing 'images' and 'sparse' subdirectories"
    ),
    sparse_subdir: str = typer.Option(
        "", help="Sparse reconstruction subdirectory (e.g., '0' for sparse/0/, empty for sparse/)"
    ),
    align_to_input_ext_scale: bool = typer.Option(
        True, help="Align prediction to input extrinsics scale"
    ),
    model_dir: str = typer.Option(DEFAULT_MODEL, help="Model directory path"),
    export_dir: str = typer.Option(DEFAULT_EXPORT_DIR, help="Export directory"),
    export_format: str = typer.Option("glb", help="Export format"),
    device: str = typer.Option(None, help="Device to use (auto-detects if not specified)"),
    use_backend: bool = typer.Option(False, help="Use backend service for inference"),
    backend_url: str = typer.Option(
        "http://localhost:8008", help="Backend URL (default: http://localhost:8008)"
    ),
    process_res: int = typer.Option(504, help="Processing resolution"),
    process_res_method: str = typer.Option(
        "upper_bound_resize", help="Processing resolution method"
    ),
    export_feat: str = typer.Option(
        "",
        help="Export features from specified layers using comma-separated indices (e.g., '0,1,2').",
    ),
    auto_cleanup: bool = typer.Option(
        False, help="Automatically clean export directory if it exists (no prompt)"
    ),
    # GLB export options
    conf_thresh_percentile: float = typer.Option(
        40.0, help="[GLB] Lower percentile for adaptive confidence threshold"
    ),
    num_max_points: int = typer.Option(
        1_000_000, help="[GLB] Maximum number of points in the point cloud"
    ),
    show_cameras: bool = typer.Option(
        True, help="[GLB] Show camera wireframes in the exported scene"
    ),
    # Feat_vis export options
    feat_vis_fps: int = typer.Option(15, help="[FEAT_VIS] Frame rate for output video"),
):
    """Run pose conditioned depth estimation on COLMAP data."""
    # Process input
    image_files, extrinsics, intrinsics = ColmapHandler.process(colmap_dir, sparse_subdir)

    # Handle export directory
    export_dir = InputHandler.handle_export_dir(export_dir, auto_cleanup)

    # Parse export_feat parameter
    export_feat_layers = parse_export_feat(export_feat)

    # Determine backend URL based on use_backend flag
    final_backend_url = backend_url if use_backend else None

    # Run inference
    run_inference(
        image_paths=image_files,
        export_dir=export_dir,
        model_dir=model_dir,
        device=device,
        backend_url=final_backend_url,
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


@app.command()
def video(
    video_path: str = typer.Argument(..., help="Path to input video file"),
    fps: float = typer.Option(1.0, help="Sampling FPS for frame extraction"),
    model_dir: str = typer.Option(DEFAULT_MODEL, help="Model directory path"),
    export_dir: str = typer.Option(DEFAULT_EXPORT_DIR, help="Export directory"),
    export_format: str = typer.Option("glb", help="Export format"),
    device: str = typer.Option(None, help="Device to use (auto-detects if not specified)"),
    use_backend: bool = typer.Option(False, help="Use backend service for inference"),
    backend_url: str = typer.Option(
        "http://localhost:8008", help="Backend URL (default: http://localhost:8008)"
    ),
    process_res: int = typer.Option(504, help="Processing resolution"),
    process_res_method: str = typer.Option(
        "upper_bound_resize", help="Processing resolution method"
    ),
    export_feat: str = typer.Option(
        "",
        help="[FEAT_VIS] Export features from specified layers using comma-separated indices (e.g., '0,1,2').",
    ),
    auto_cleanup: bool = typer.Option(
        False, help="Automatically clean export directory if it exists (no prompt)"
    ),
    # GLB export options
    conf_thresh_percentile: float = typer.Option(
        40.0, help="[GLB] Lower percentile for adaptive confidence threshold"
    ),
    num_max_points: int = typer.Option(
        1_000_000, help="[GLB] Maximum number of points in the point cloud"
    ),
    show_cameras: bool = typer.Option(
        True, help="[GLB] Show camera wireframes in the exported scene"
    ),
    # Feat_vis export options
    feat_vis_fps: int = typer.Option(15, help="[FEAT_VIS] Frame rate for output video"),
):
    """Run depth estimation on video by extracting frames and processing them."""
    # Handle export directory
    export_dir = InputHandler.handle_export_dir(export_dir, auto_cleanup)

    # Process input
    image_files = VideoHandler.process(video_path, export_dir, fps)

    # Parse export_feat parameter
    export_feat_layers = parse_export_feat(export_feat)

    # Determine backend URL based on use_backend flag
    final_backend_url = backend_url if use_backend else None

    # Run inference
    run_inference(
        image_paths=image_files,
        export_dir=export_dir,
        model_dir=model_dir,
        device=device,
        backend_url=final_backend_url,
        export_format=export_format,
        process_res=process_res,
        process_res_method=process_res_method,
        export_feat_layers=export_feat_layers,
        conf_thresh_percentile=conf_thresh_percentile,
        num_max_points=num_max_points,
        show_cameras=show_cameras,
        feat_vis_fps=feat_vis_fps,
    )


# ============================================================================
# Service management commands
# ============================================================================


@app.command()
def backend(
    model_dir: str = typer.Option(DEFAULT_MODEL, help="Model directory path"),
    device: str = typer.Option(None, help="Device to use (auto-detects if not specified)"),
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(8008, help="Port to bind to"),
    gallery_dir: str = typer.Option(DEFAULT_GALLERY_DIR, help="Gallery directory path (optional)"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
):
    """Start model backend service with integrated gallery."""
    import logging

    # Configure logging
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    numeric_level = log_level_map.get(log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    typer.echo("=" * 60)
    typer.echo("🚀 Starting Depth Anything 3 Backend Server")
    typer.echo("=" * 60)
    typer.echo(f"Model directory: {model_dir}")
    typer.echo(f"Device: {device}")
    typer.echo(f"Log level: {log_level.upper()}")

    # Check if gallery directory exists
    if gallery_dir and os.path.exists(gallery_dir):
        typer.echo(f"Gallery directory: {gallery_dir}")
    else:
        gallery_dir = None  # Disable gallery if directory doesn't exist

    typer.echo()
    typer.echo("📡 Server URLs (Ctrl/CMD+Click to open):")
    typer.echo(f"  🏠 Home:      http://{host}:{port}")
    typer.echo(f"  📊 Dashboard: http://{host}:{port}/dashboard")
    typer.echo(f"  📈 API Status: http://{host}:{port}/status")

    if gallery_dir:
        typer.echo(f"  🎨 Gallery:   http://{host}:{port}/gallery/")

    typer.echo("=" * 60)

    try:
        start_server(model_dir, device, host, port, gallery_dir)
    except KeyboardInterrupt:
        typer.echo("\n👋 Backend server stopped.")
    except Exception as e:
        typer.echo(f"❌ Failed to start backend: {e}")
        raise typer.Exit(1)


# ============================================================================
# Application launch commands
# ============================================================================


@app.command()
def gradio(
    model_dir: str = typer.Option(DEFAULT_MODEL, help="Model directory path"),
    workspace_dir: str = typer.Option(DEFAULT_GRADIO_DIR, help="Workspace directory path"),
    gallery_dir: str = typer.Option(DEFAULT_GALLERY_DIR, help="Gallery directory path"),
    host: str = typer.Option("127.0.0.1", help="Host address to bind to"),
    port: int = typer.Option(7860, help="Port number to bind to"),
    share: bool = typer.Option(False, help="Create a public link for the app"),
    debug: bool = typer.Option(False, help="Enable debug mode"),
    cache_examples: bool = typer.Option(
        False, help="Pre-cache all example scenes at startup for faster loading"
    ),
    cache_gs_tag: str = typer.Option(
        "",
        help="Tag to match scene names for high-res+3DGS caching (e.g., 'dl3dv'). Scenes containing this tag will use high_res and infer_gs=True; others will use low_res only.",
    ),
):
    """Launch Depth Anything 3 Gradio interactive web application"""
    from depth_anything_3.app.gradio_app import DepthAnything3App

    # Create necessary directories
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(gallery_dir, exist_ok=True)

    typer.echo("Launching Depth Anything 3 Gradio application...")
    typer.echo(f"Model directory: {model_dir}")
    typer.echo(f"Workspace directory: {workspace_dir}")
    typer.echo(f"Gallery directory: {gallery_dir}")
    typer.echo(f"Host: {host}")
    typer.echo(f"Port: {port}")
    typer.echo(f"Share: {share}")
    typer.echo(f"Debug mode: {debug}")
    typer.echo(f"Cache examples: {cache_examples}")
    if cache_examples:
        if cache_gs_tag:
            typer.echo(
                f"Cache GS Tag: '{cache_gs_tag}' (scenes matching this tag will use high-res + 3DGS)"
            )
        else:
            typer.echo(f"Cache GS Tag: None (all scenes will use low-res only)")

    try:
        # Initialize and launch application
        app = DepthAnything3App(
            model_dir=model_dir, workspace_dir=workspace_dir, gallery_dir=gallery_dir
        )

        # Pre-cache examples if requested
        if cache_examples:
            typer.echo("\n" + "=" * 60)
            typer.echo("Pre-caching mode enabled")
            if cache_gs_tag:
                typer.echo(f"Scenes containing '{cache_gs_tag}' will use HIGH-RES + 3DGS")
                typer.echo(f"Other scenes will use LOW-RES only")
            else:
                typer.echo(f"All scenes will use LOW-RES only")
            typer.echo("=" * 60)
            app.cache_examples(
                show_cam=True,
                filter_black_bg=False,
                filter_white_bg=False,
                save_percentage=20.0,
                num_max_points=1000,
                cache_gs_tag=cache_gs_tag,
                gs_trj_mode="smooth",
                gs_video_quality="low",
            )

        # Prepare launch arguments
        launch_kwargs = {"share": share, "debug": debug}

        app.launch(host=host, port=port, **launch_kwargs)

    except KeyboardInterrupt:
        typer.echo("\nGradio application stopped.")
    except Exception as e:
        typer.echo(f"Failed to launch Gradio application: {e}")
        raise typer.Exit(1)


@app.command()
def gallery(
    gallery_dir: str = typer.Option(DEFAULT_GALLERY_DIR, help="Gallery root directory"),
    host: str = typer.Option("127.0.0.1", help="Host address to bind to"),
    port: int = typer.Option(8007, help="Port number to bind to"),
    open_browser: bool = typer.Option(False, help="Open browser after launch"),
):
    """Launch Depth Anything 3 Gallery server"""

    # Validate gallery directory
    if not os.path.exists(gallery_dir):
        raise typer.BadParameter(f"Gallery directory not found: {gallery_dir}")

    typer.echo("Launching Depth Anything 3 Gallery server...")
    typer.echo(f"Gallery directory: {gallery_dir}")
    typer.echo(f"Host: {host}")
    typer.echo(f"Port: {port}")
    typer.echo(f"Auto-open browser: {open_browser}")

    try:
        # Set command line arguments
        import sys

        sys.argv = ["gallery", "--dir", gallery_dir, "--host", host, "--port", str(port)]
        if open_browser:
            sys.argv.append("--open")

        # Launch gallery server
        gallery_main()

    except KeyboardInterrupt:
        typer.echo("\nGallery server stopped.")
    except Exception as e:
        typer.echo(f"Failed to launch Gallery server: {e}")
        raise typer.Exit(1)


@app.command()
def benchmark(
    config: str = typer.Option(None, help="Path to benchmark YAML config file"),
    output_dir: str = typer.Option("benchmark_results", help="Output directory for results"),
    input_video: str = typer.Option(None, help="Path to test video (overrides config)"),
    device: str = typer.Option(None, help="Device to use (overrides config)"),
    create_default: bool = typer.Option(False, help="Create default benchmark config and exit"),
    num_frames: int = typer.Option(100, help="Number of frames to test (for default config)"),
    use_backend: bool = typer.Option(False, help="Use backend service for inference"),
    backend_url: str = typer.Option("http://localhost:8008", help="Backend service URL"),
    start_backend: bool = typer.Option(False, help="Auto-start backend service"),
    stop_backend: bool = typer.Option(False, help="Auto-stop backend service after completion"),
):
    """
    Run streaming benchmarks to compare models and configurations.

    This command benchmarks different model sizes (DA3-SMALL, DA3-BASE, DA3-LARGE)
    with various configurations (resolutions, precisions, multi-camera setups) and
    generates comprehensive reports with performance metrics and visualizations.

    Example usage:
        # Create default benchmark config
        da3 benchmark --create-default --output-dir my_benchmark

        # Run benchmark with custom config
        da3 benchmark --config benchmark_config.yaml

        # Run default benchmark
        da3 benchmark --input-video test.mp4 --device mps --num-frames 100
    """
    from pathlib import Path
    from depth_anything_3.benchmarking import (
        BenchmarkConfig,
        BenchmarkRunner,
        ReportGenerator,
        create_default_benchmark_config,
    )

    # Handle create-default flag
    if create_default:
        output_path = Path(output_dir) / "benchmark_config.yaml"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Auto-detect device
        detected_device = device or get_default_device()

        # Create default config
        default_config = create_default_benchmark_config(
            name="Default Streaming Benchmark",
            input_video=Path(input_video) if input_video else None,
            device=detected_device,
        )

        # Save to YAML
        default_config.save_to_yaml(output_path)
        typer.echo(f"✅ Created default benchmark config: {output_path}")
        typer.echo(f"\nEdit the config file and run:")
        typer.echo(f"  da3 benchmark --config {output_path}")
        return

    # Load config from file or create default
    if config:
        typer.echo(f"📋 Loading benchmark config from: {config}")
        benchmark_config = BenchmarkConfig.from_yaml(Path(config))
    else:
        typer.echo("📋 No config provided, creating default benchmark")
        detected_device = device or get_default_device()
        benchmark_config = create_default_benchmark_config(
            name="Quick Streaming Benchmark",
            input_video=Path(input_video) if input_video else None,
            device=detected_device,
        )

        # Update output dir if specified
        if output_dir != "benchmark_results":
            benchmark_config.output_dir = Path(output_dir)

        # Update num_frames for all scenarios
        for scenario in benchmark_config.scenarios:
            scenario.num_frames = num_frames

    # Override device if specified
    if device:
        for scenario in benchmark_config.scenarios:
            scenario.device = device

    # Override input video if specified
    if input_video:
        benchmark_config.input_video = Path(input_video)

    # Override backend settings if specified
    if use_backend:
        benchmark_config.use_backend = True
        benchmark_config.backend_url = backend_url
        benchmark_config.start_backend = start_backend
        benchmark_config.stop_backend = stop_backend

    # Display config
    typer.echo("\n" + "=" * 70)
    typer.echo(f"🚀 {benchmark_config.name}")
    typer.echo("=" * 70)
    if benchmark_config.description:
        typer.echo(f"📝 {benchmark_config.description}")
    typer.echo(f"📁 Output: {benchmark_config.output_dir}")
    if benchmark_config.input_video:
        typer.echo(f"🎥 Input video: {benchmark_config.input_video}")
    if benchmark_config.use_backend:
        typer.echo(f"🔌 Backend: {benchmark_config.backend_url}")
        if benchmark_config.start_backend:
            typer.echo(f"   Auto-start: ✅")
        if benchmark_config.stop_backend:
            typer.echo(f"   Auto-stop: ✅")
    typer.echo(f"📊 Scenarios: {len(benchmark_config.scenarios)}")
    for scenario in benchmark_config.scenarios:
        typer.echo(f"   - {scenario.name}")
    typer.echo("=" * 70)
    typer.echo()

    # Confirm before starting
    if not typer.confirm("Start benchmark?"):
        typer.echo("❌ Benchmark cancelled")
        raise typer.Exit(0)

    # Run benchmark
    try:
        runner = BenchmarkRunner(benchmark_config)
        results = runner.run_all()

        # Generate report
        typer.echo("\n" + "=" * 70)
        typer.echo("📊 Generating HTML report...")
        typer.echo("=" * 70)

        report_gen = ReportGenerator(benchmark_config.output_dir)
        report_path = report_gen.generate(
            results,
            benchmark_config.name,
            benchmark_config.description,
        )

        # Display summary
        typer.echo("\n" + "=" * 70)
        typer.echo("✅ Benchmark Complete!")
        typer.echo("=" * 70)
        typer.echo(f"📄 HTML Report: {report_path}")
        typer.echo(f"📁 Results: {benchmark_config.output_dir}")
        typer.echo()

        # Display quick summary
        typer.echo("Quick Summary:")
        for result in results:
            typer.echo(f"  {result.scenario_name}:")
            typer.echo(f"    - FPS: {result.avg_fps:.2f} (min: {result.min_fps:.2f}, max: {result.max_fps:.2f})")
            typer.echo(f"    - Latency: {result.avg_latency_ms:.2f}ms (p95: {result.p95_latency_ms:.2f}ms)")
            typer.echo(f"    - Memory: {result.avg_memory_mb:.2f}MB (peak: {result.max_memory_mb:.2f}MB)")
            typer.echo()

        typer.echo(f"🌐 Open report in browser:")
        typer.echo(f"  open {report_path}")

    except KeyboardInterrupt:
        typer.echo("\n⏸️  Benchmark interrupted")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def stream(
    model_dir: str = typer.Option(DEFAULT_MODEL, help="Model directory path"),
    device: str = typer.Option(None, help="Device to use (auto-detects if not specified)"),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8080, help="Port to bind to"),
    process_res: int = typer.Option(504, help="Processing resolution (rounded to nearest multiple of 14)"),
    window_size: int = typer.Option(None, help="Default window size (frames per processing window)"),
    overlap: int = typer.Option(None, help="Default overlap between windows"),
    buffer_size: int = typer.Option(None, help="Default buffer size before processing"),
    max_fps: float = typer.Option(None, help="Default max FPS limit"),
    quality: int = typer.Option(85, help="Default JPEG quality for depth output (0-100)"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
):
    """
    Start real-time streaming server for depth estimation.

    Provides HTTP and WebSocket endpoints for real-time depth estimation,
    optimized for integration with TouchDesigner, vvvv, Processing, etc.

    Endpoints:
        - POST /process/frame: Process single frame via HTTP
        - WS /stream: Real-time WebSocket streaming
        - GET /stats: Get streaming statistics
        - GET /health: Health check

    Example (TouchDesigner):
        1. Connect to ws://localhost:8080/stream
        2. Send JPEG frames via WebSocket
        3. Receive depth maps in real-time
    """
    import logging

    # Configure logging
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    numeric_level = log_level_map.get(log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    typer.echo("=" * 60)
    typer.echo("🌊 Starting Depth Anything 3 Streaming Server")
    typer.echo("=" * 60)
    typer.echo(f"Model directory: {model_dir}")
    typer.echo(f"Log level: {log_level.upper()}")

    # Auto-detect device if not specified
    if device is None:
        device = get_default_device()
        typer.echo(f"Device: {device} (auto-detected)")
    else:
        typer.echo(f"Device: {device}")

    typer.echo()
    typer.echo("📡 Server will be available at:")
    typer.echo(f"  🏠 Root:        http://{host}:{port}")
    typer.echo(f"  📊 Health:      http://{host}:{port}/health")
    typer.echo(f"  📈 Stats:       http://{host}:{port}/stats")
    typer.echo(f"  🔗 HTTP Frame:  http://{host}:{port}/process/frame")
    typer.echo(f"  🌐 WebSocket:   ws://{host}:{port}/stream")
    typer.echo("=" * 60)
    typer.echo()
    typer.echo("💡 TouchDesigner Integration:")
    typer.echo("   1. Add WebSocket DAT to your network")
    typer.echo(f"   2. Set Address: ws://localhost:{port}/stream")
    typer.echo("   3. Send frames: webSocketDAT.sendBinary(jpeg_bytes)")
    typer.echo("   4. Receive via callbacks DAT onReceiveText()")
    typer.echo()
    typer.echo("⚙️  URL Parameters (customize per connection):")
    typer.echo(f"   ws://localhost:{port}/stream?window_size=8&max_fps=15&quality=80")
    typer.echo("   - window_size: Frames per window (lower=faster, higher=smoother)")
    typer.echo("   - overlap: Overlapping frames (default: window_size/3)")
    typer.echo("   - buffer_size: Min frames before processing (default: window_size/2)")
    typer.echo("   - max_fps: Throttle processing FPS")
    typer.echo("   - quality: JPEG quality 0-100 (lower=faster)")
    typer.echo()

    try:
        start_stream_server(
            model_dir=model_dir,
            device=device,
            host=host,
            port=port,
            process_res=process_res,
        )
    except KeyboardInterrupt:
        typer.echo("\n👋 Streaming server stopped.")
    except Exception as e:
        typer.echo(f"❌ Failed to start streaming server: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
