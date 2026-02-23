# Depth Anything 3 - Streaming Benchmarks

This directory contains example benchmark configurations for testing streaming performance.

## Overview

The benchmarking system allows you to:

- **Compare model sizes**: DA3-SMALL, DA3-BASE, DA3-LARGE
- **Test different configurations**: Resolutions, precisions (fp16/fp32), multi-camera setups
- **Collect comprehensive metrics**: FPS, latency, memory usage, frame-level statistics
- **Generate visual reports**: HTML reports with interactive charts, comparison videos, depth map exports

## Quick Start

### 1. Run Quick Benchmark (Recommended First Step)

Test all three models with default settings (~15 minutes):

```bash
da3 benchmark --config benchmarks/example_quick.yaml
```

This will:
- Test DA3-SMALL, DA3-BASE, and DA3-LARGE
- Process 50 frames per model
- Generate comparison videos and depth maps
- Create an HTML report with performance charts

### 2. Run Comprehensive Benchmark

Full test across resolutions, precisions, and multi-camera setups (~1-2 hours):

```bash
da3 benchmark --config benchmarks/example_comprehensive.yaml
```

### 3. Quick Test Without Config

Test with your own video:

```bash
da3 benchmark --input-video path/to/video.mp4 --device mps --num-frames 50
```

### 4. Create Custom Config

Generate a default config template to customize:

```bash
da3 benchmark --create-default --output-dir my_benchmark
```

This creates `my_benchmark/benchmark_config.yaml` which you can edit and run:

```bash
da3 benchmark --config my_benchmark/benchmark_config.yaml
```

## Example Configurations

### [example_quick.yaml](example_quick.yaml)

**Purpose**: Quick performance comparison of all models
**Time**: ~15 minutes on Apple Silicon
**Scenarios**: 3 (SMALL, BASE, LARGE with fp32)
**Frames**: 50 per scenario

```bash
da3 benchmark --config benchmarks/example_quick.yaml
```

### [example_comprehensive.yaml](example_comprehensive.yaml)

**Purpose**: Full benchmark with all configurations
**Time**: ~1-2 hours on Apple Silicon
**Scenarios**: 13 (all models × resolutions × precisions × cameras)
**Frames**: 100 per scenario

```bash
da3 benchmark --config benchmarks/example_comprehensive.yaml
```

## Understanding Results

### HTML Report

The benchmark generates an interactive HTML report at `benchmark_results/report.html` with:

1. **Performance Summary Table**
   - Quick comparison of all scenarios
   - FPS, latency, and memory metrics

2. **Interactive Charts**
   - Average FPS comparison
   - Latency (avg and p95) comparison
   - Memory usage comparison
   - Frame-by-frame processing time

3. **Detailed Results**
   - Per-scenario statistics
   - Min/max FPS, percentile latencies
   - Memory peak usage

### Output Structure

```
benchmark_results/
├── report.html                          # Main HTML report
├── DA3-SMALL_1cam_640x480_fp32/
│   ├── metrics.json                     # Raw metrics
│   ├── comparison.mp4                   # Side-by-side video
│   └── depth_maps/                      # Individual depth maps
│       ├── depth_0000.png
│       ├── depth_0001.png
│       └── ...
├── DA3-BASE_1cam_640x480_fp32/
│   └── ...
└── DA3-LARGE_1cam_640x480_fp32/
    └── ...
```

## Configuration Reference

### Scenario Parameters

```yaml
scenarios:
  - name: "DA3-BASE_1cam_640x480_fp32"    # Unique identifier

    # Model configuration
    model:
      name: "DA3-BASE"
      model_dir: "depth-anything/DA3-BASE"

    # Input configuration
    num_cameras: 1                         # 1, 2, or 3
    resolution: [640, 480]                 # [width, height]

    # Processing configuration
    precision: "fp32"                      # "fp16" or "fp32"
    device: "mps"                          # "mps", "cuda", or "cpu"
    process_res: 504                       # Internal processing resolution

    # Streaming configuration
    stream_config:
      window_size: null                    # null = auto-detect
      overlap: null                        # null = auto-detect
      buffer_size: null                    # null = auto-detect
      output_latency_frames: 2             # Output buffering

    # Test configuration
    num_frames: 100                        # Frames to process
    warmup_frames: 10                      # Warmup (excluded from metrics)

    # Output configuration
    save_depth_maps: true                  # Save individual depth maps
    save_comparison_video: true            # Create comparison video
```

### Global Parameters

```yaml
name: "My Benchmark"                       # Benchmark name
description: "..."                         # Optional description

input_video: "path/to/video.mp4"          # Test video (optional)
input_images: "path/to/images/"           # Test images (optional)

output_dir: "benchmark_results"           # Output directory

compare_against_baseline: true            # Enable comparison
baseline_scenario_name: "DA3-BASE_..."    # Reference scenario
```

## Tips for Effective Benchmarking

### 1. Start Small

Always run `example_quick.yaml` first to:
- Verify setup is working
- Get baseline performance expectations
- Identify any issues before long runs

### 2. Use Real Test Data

For accurate results, provide your own test video:

```yaml
input_video: "path/to/your_video.mp4"
```

Without this, synthetic frames are generated (less realistic).

### 3. Choose Frame Count Wisely

- **Quick test**: 50 frames (~5 minutes per scenario)
- **Standard test**: 100 frames (~10 minutes per scenario)
- **Thorough test**: 200+ frames (for statistical significance)

### 4. Platform-Specific Settings

**macOS (Apple Silicon)**:
```yaml
device: "mps"
precision: "fp32"  # fp16 NOT supported on MPS - will auto-fallback to fp32
```

**Linux/Windows (NVIDIA GPU)**:
```yaml
device: "cuda"
precision: "fp16"  # Significant speedup on modern GPUs (RTX 30/40 series)
precision: "fp32"  # For older GPUs or maximum accuracy
```

**⚠️ Important**: fp16 precision is **only supported on CUDA**. On MPS/CPU, fp16 will automatically fall back to fp32 with a warning.

### 5. Memory Considerations

Multi-camera and high-resolution tests use more memory:

- **1 camera @ 640×480**: ~2GB VRAM
- **2 cameras @ 640×480**: ~4GB VRAM
- **3 cameras @ 640×480**: ~6GB VRAM
- **1 camera @ 1280×720**: ~3GB VRAM

If you encounter OOM errors, reduce:
- `num_cameras`
- `resolution`
- `num_frames`

## Interpreting Metrics

### FPS (Frames Per Second)

- **Higher is better**
- Streaming target: ≥15 FPS for real-time
- Expected on MPS:
  - DA3-SMALL: 20-30 FPS
  - DA3-BASE: 10-20 FPS
  - DA3-LARGE: 5-10 FPS

### Latency (milliseconds)

- **Lower is better**
- Total time from input to output
- **p95/p99**: Worst-case scenarios (important for consistency)
- Real-time target: <100ms

### Memory Usage

- **Peak memory**: Maximum GPU/MPS memory used
- **Average memory**: Typical usage during processing
- Important for:
  - Multi-tasking (running other apps)
  - Determining max workload capacity

## Example Workflows

### Compare Model Sizes

```bash
# Quick test to choose best model for your use case
da3 benchmark --config benchmarks/example_quick.yaml
```

### Optimize for Real-Time Performance

1. Start with quick benchmark
2. Note FPS for each model
3. Test fp16 for CUDA GPUs:

```yaml
precision: "fp16"
device: "cuda"
```

4. Adjust resolution if needed

### Test Multi-Camera Setup

```yaml
scenarios:
  - name: "MultiCam_Test"
    model:
      name: "DA3-BASE"
      model_dir: "depth-anything/DA3-BASE"
    num_cameras: 2  # or 3
    resolution: [640, 480]
    num_frames: 100
```

### Benchmark Your Hardware

Create a config with your typical use case:

```yaml
name: "My Hardware Benchmark"
input_video: "my_typical_video.mp4"

scenarios:
  - name: "MyUseCase"
    model:
      name: "DA3-BASE"
      model_dir: "depth-anything/DA3-BASE"
    num_cameras: 1
    resolution: [1280, 720]  # Your target resolution
    precision: "fp32"
    device: "mps"  # Your device
    num_frames: 200  # Longer run for accuracy
```

## Troubleshooting

### OOM (Out of Memory)

**Symptom**: Crash with memory allocation error

**Solutions**:
1. Reduce `num_cameras`
2. Lower `resolution`
3. Use smaller model (SMALL instead of BASE/LARGE)
4. Reduce `num_frames`

### Slow Performance

**Symptom**: Much lower FPS than expected

**Checks**:
1. Ensure device is correct (`mps` for Apple Silicon, `cuda` for NVIDIA)
2. Close other GPU-intensive applications
3. Check system thermal throttling
4. Verify model is actually loaded on GPU (check metrics)

### Missing Depth Maps

**Symptom**: No depth maps saved

**Solution**: Enable in scenario config:
```yaml
save_depth_maps: true
save_comparison_video: true
```

## Next Steps

After benchmarking:

1. **Review HTML report**: Open `benchmark_results/report.html`
2. **Compare scenarios**: Look at FPS and latency charts
3. **Watch comparison videos**: Visual quality assessment
4. **Choose configuration**: Based on FPS/quality tradeoff
5. **Integrate into pipeline**: Use settings in your application

## Questions?

See the main [CLAUDE.md](../CLAUDE.md) for detailed streaming architecture documentation.
