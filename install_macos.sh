#!/bin/bash
# Installation script for Depth Anything 3 on macOS with MPS support

set -e  # Exit on error

echo "🍎 Installing Depth Anything 3 for macOS with MPS support..."
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Not in a virtual environment. Activate .venv first:"
    echo "   source .venv/bin/activate"
    exit 1
fi

echo "✓ Virtual environment detected: $VIRTUAL_ENV"

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: This script is designed for macOS."
    echo "   For Linux/Windows, use: pip install -e \".[all]\""
    exit 1
fi

echo "✓ Running on macOS"

# Step 1: Install base dependencies (including PyTorch)
echo ""
echo "📦 Step 1/3: Installing base dependencies..."
pip install -e .

# Step 2: Install Gradio app dependencies
echo ""
echo "📦 Step 2/3: Installing Gradio dependencies..."
pip install "gradio>=5" "pillow>=9.0"

# Step 3: Install gsplat-mps (requires PyTorch to be already installed)
echo ""
echo "📦 Step 3/3: Installing gsplat-mps for Gaussian Splatting..."
pip install --no-build-isolation git+https://github.com/iffyloop/gsplat-mps.git

echo ""
echo "🔍 Verifying installation..."

# Test gsplat
if python -c "import gsplat; from gsplat import rasterization" 2>/dev/null; then
    echo "✓ gsplat (MPS version) installed successfully"
else
    echo "❌ Error: gsplat import failed"
    exit 1
fi

# Test depth_anything_3
if python -c "import sys; sys.path.insert(0, 'src'); from depth_anything_3.api import DepthAnything3" 2>/dev/null; then
    echo "✓ Depth Anything 3 installed successfully"
else
    echo "⚠️  Note: depth_anything_3 package might need reinstallation"
    echo "   Attempting to fix..."
    pip install -e . --force-reinstall --no-deps
    if python -c "from depth_anything_3.api import DepthAnything3" 2>/dev/null; then
        echo "✓ Depth Anything 3 installed successfully (after fix)"
    else
        echo "❌ Error: depth_anything_3 import still failing"
        echo "   You can still use it via: python -m depth_anything_3.cli"
    fi
fi

echo ""
echo "✅ Installation complete! All dependencies verified."
echo ""
echo "🚀 Quick start:"
echo "   # Start backend with MPS"
echo "   da3 backend --model-dir depth-anything/DA3NESTED-GIANT-LARGE --device mps"
echo ""
echo "   # Process a video"
echo "   da3 video path/to/video.mp4 --device mps --export-dir output"
echo ""
