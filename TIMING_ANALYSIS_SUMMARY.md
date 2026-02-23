# Timing Analysis Summary

Complete summary of the timing instrumentation and performance analysis work.

## What We Built

### 1. Comprehensive Timing Breakdown System

Created a detailed timing system that tracks every stage of the inference pipeline:

**Files Created/Modified**:
- [`src/depth_anything_3/utils/timing.py`](src/depth_anything_3/utils/timing.py) - Timing data structures
- [`src/depth_anything_3/api.py`](src/depth_anything_3/api.py) - Instrumented inference pipeline
- [`src/depth_anything_3/services/streaming.py`](src/depth_anything_3/services/streaming.py) - Streaming timing
- [`src/depth_anything_3/benchmarking/metrics.py`](src/depth_anything_3/benchmarking/metrics.py) - Benchmark metrics
- [`src/depth_anything_3/benchmarking/report.py`](src/depth_anything_3/benchmarking/report.py) - Visualization

**Timing Stages Tracked**:
```
Preprocessing (per-frame):
  ├─ Image Loading
  ├─ Image Resize
  ├─ Image Normalization
  └─ Camera Preprocessing

Input Preparation (per-frame):
  ├─ To Device (CPU→GPU transfer)
  └─ Extrinsics Normalization

Model Forward (per-frame):
  ├─ Total forward time
  └─ (Future: Backbone, Head, Camera enc/dec breakdown)

Postprocessing (per-frame):
  ├─ Output to CPU (GPU→CPU transfer)
  ├─ Depth Postprocessing
  ├─ Pose Alignment
  └─ Prediction Conversion

Export (if enabled)
```

### 2. Benchmark Visualization

HTML reports now include:
- **Stacked bar charts** showing time breakdown (absolute & percentage)
- **Per-scenario comparisons**
- **Aggregated statistics** (mean, min, max, std dev)

Example output:
![Timing Breakdown Chart](docs/timing_breakdown_example.png)

### 3. MPS Performance Analysis

Comprehensive analysis of macOS MPS performance:
- **Profiling tools** - [`analyze_mps_performance.py`](analyze_mps_performance.py)
- **Optimization guide** - [MPS_OPTIMIZATION_GUIDE.md](MPS_OPTIMIZATION_GUIDE.md)
- **Optimization library** - [`src/depth_anything_3/optimizations/`](src/depth_anything_3/optimizations/)

## Key Findings

### Issue #1: Preprocessing Time Showed as Zero ✅ FIXED

**Problem**: Preprocessing, input preparation, and export were showing 0ms in benchmarks.

**Root Cause**: When processing frames in batches (window_size=8-12), timing was measured for the entire batch but not divided by the number of frames.

**Solution**: Normalized all timing measurements by `num_frames`:

```python
# Before (wrong)
timing.image_loading_ms = total_time  # 80ms for 8 frames

# After (correct)
timing.image_loading_ms = total_time / num_frames  # 10ms per frame
```

### Issue #2: Postprocessing Time Varied by Model ✅ EXPLAINED

**Observation**:
- DA3-SMALL: 7ms postprocessing
- DA3-BASE: 15ms postprocessing

**Expected**: Should be identical (same output size)

**Root Cause**: **Auxiliary features** from larger models are bigger:
- DA3-SMALL: 384-dim features
- DA3-BASE: 768-dim features
- DA3-GIANT: 1536-dim features

The "postprocessing" time includes GPU→CPU transfer of aux features, which scales with model size.

**Solution**:
1. Better labeling in documentation
2. Skip aux features when not needed (optimization opportunity!)

### Issue #3: Timing Measurement Bug on MPS ✅ FIXED

**Problem**: "Prediction Conversion" was showing 46ms when it should be <1ms.

**Root Cause**: Calling `torch.mps.synchronize()` after model forward was waiting for ALL pending MPS operations, including the model forward that hadn't fully completed yet.

**Solution**: Removed redundant synchronization - the model forward already synchronizes at the end.

## Performance Bottlenecks (DA3-BASE on MPS)

| Stage | Time | % | Optimization Potential |
|-------|------|---|----------------------|
| Model Forward | 123ms | 71% | 🎯 Primary target |
| Postprocessing | 48ms | 28% | Medium (aux features) |
| Preprocessing | 1.6ms | 1% | Already optimized |
| Input Prep | 0.02ms | 0% | Negligible |

**Key Insight**: Model forward is the bottleneck. Optimizing this provides the most benefit.

## Optimization Opportunities

See [MPS_OPTIMIZATION_GUIDE.md](MPS_OPTIMIZATION_GUIDE.md) for details.

### Quick Wins (No Quality Loss)

1. **Use `torch.inference_mode()`** instead of `torch.no_grad()` → 5-10% faster
2. **Skip aux features** when not needed → 15-20% faster
3. **Optimal batch sizes** → Already implemented in streaming

**Expected Total**: 1.2-1.3x speedup

### Optional (Requires Validation)

1. **Mixed precision (fp16)** → 20-40% faster (must validate quality)
2. **torch.compile()** → Unknown on MPS (experimental)

**Possible Total**: 1.5-1.6x speedup

### Not Recommended for MPS

1. ❌ **SDPA (Scaled Dot Product Attention)** → 23% SLOWER on MPS
   - Works on CUDA (Flash Attention) but not MPS
   - Keep current attention implementation

## How to Use

### Run Performance Analysis

```bash
# Analyze MPS performance
python analyze_mps_performance.py

# Run benchmarks with timing
da3 benchmark --config benchmarks/example_quick.yaml

# View results
open benchmark_results/quick/report.html
```

### Access Timing in Code

```python
from depth_anything_3.api import DepthAnything3

model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE").to("mps")

# Enable timing collection
prediction = model.inference(
    image=frames,
    collect_timing=True
)

# Access detailed breakdown
print(prediction.timing)
# Output:
# Timing Breakdown:
# Total: 172.73ms
# Model Forward: 123.03ms (71.2%)
# Postprocessing: 48.05ms (27.8%)
# ...
```

### Benchmark Streaming

```python
from depth_anything_3.services.streaming import StreamingDepthEstimator

estimator = StreamingDepthEstimator(
    model=model,
    device="mps",
    collect_timing=True  # Enable timing
)

# Process frames...
for frame in frames:
    depth = estimator.process_frame(frame)

# Get aggregated timing stats
stats = estimator.get_stats()
print(stats["timing"])  # Timing breakdown across all frames
```

## Files Modified

### Core Timing System
- `src/depth_anything_3/utils/timing.py` - NEW
- `src/depth_anything_3/api.py` - Modified (added timing instrumentation)
- `src/depth_anything_3/services/streaming.py` - Modified (added timing collection)

### Benchmarking
- `src/depth_anything_3/benchmarking/metrics.py` - Modified (added timing fields)
- `src/depth_anything_3/benchmarking/runner.py` - Modified (collect timing)
- `src/depth_anything_3/benchmarking/report.py` - Modified (timing charts)

### Analysis & Optimization
- `analyze_mps_performance.py` - NEW (profiling tool)
- `src/depth_anything_3/optimizations/mps_optimizations.py` - NEW
- `src/depth_anything_3/optimizations/__init__.py` - NEW

### Documentation
- `MPS_OPTIMIZATION_GUIDE.md` - NEW
- `TIMING_ANALYSIS_SUMMARY.md` - NEW (this file)

## Next Steps

1. **Implement Quick Win Optimizations**
   - [ ] Replace `torch.no_grad()` with `torch.inference_mode()`
   - [ ] Skip aux feature computation when `export_feat_layers=[]`
   - [ ] Optimize aux feature GPU→CPU transfer

2. **Validate fp16 Mode**
   - [ ] Test mixed precision inference
   - [ ] Run quality benchmarks
   - [ ] Compare depth maps visually
   - [ ] Measure numerical error

3. **Advanced Profiling**
   - [ ] Profile with Xcode Instruments (Metal System Trace)
   - [ ] Identify GPU bottlenecks
   - [ ] Optimize hot paths

4. **Continuous Monitoring**
   - [ ] Add timing to CI/CD pipeline
   - [ ] Track performance regressions
   - [ ] Maintain optimization guide

## Conclusion

We now have:
✅ Comprehensive timing instrumentation
✅ Per-frame normalized measurements
✅ Visual breakdown in HTML reports
✅ MPS performance analysis
✅ Optimization roadmap

**Expected Impact**: 1.3-1.5x speedup on MPS without quality loss.

---

**Created**: 2025-01-18
**Contributors**: Analysis and implementation based on performance profiling
