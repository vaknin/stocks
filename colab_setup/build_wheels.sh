#!/bin/bash
# Fixed Build and cache wheels for Google Colab - NO TIMEOUT
# This script builds all wheels once and saves them for fast reuse
# Expected time: 60-120 minutes (one time only)

set -e  # Exit on any error

echo "🏗️ BUILDING WHEELS FOR GOOGLE COLAB CACHE (FIXED VERSION)"
echo "=========================================================="
echo "⏱️ This will take 60-120 minutes but only needs to be done ONCE"
echo "🔄 Subsequent installs will be 2-3 minutes using cached wheels"
echo ""

# Configuration
WHEEL_CACHE_DIR="/content/drive/MyDrive/colab_wheels"
TEMP_WHEEL_DIR="/tmp/colab_wheels_build"
REQUIREMENTS_FILE="/content/colab_setup/colab_requirements.txt"

# Create directories
echo "📁 Creating wheel cache directories..."
mkdir -p "$WHEEL_CACHE_DIR"
mkdir -p "$TEMP_WHEEL_DIR"

# Verify requirements file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "❌ Requirements file not found: $REQUIREMENTS_FILE"
    exit 1
fi

echo "✅ Requirements file found: $REQUIREMENTS_FILE"
echo "💾 Wheel cache directory: $WHEEL_CACHE_DIR"
echo ""

# CRITICAL FIX: Install system dependencies first
echo "🔧 Installing system dependencies to fix setup.py egg_info errors..."
apt-get update -qq
apt-get install -y \
    build-essential \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    pkg-config \
    cmake \
    ninja-build \
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    git

# CRITICAL FIX: Upgrade pip, setuptools, and wheel first
echo "📦 Upgrading pip, setuptools, and wheel..."
python3 -m pip install --upgrade pip setuptools wheel

# CRITICAL FIX: Install numpy first (many packages need it for setup.py)
echo "🔢 Installing numpy first to fix dependency issues..."
python3 -m pip install "numpy==2.2.6"

echo "✅ System dependencies and numpy installed"
echo ""

# Function to keep Colab alive during long builds
keep_alive() {
    while true; do
        sleep 300  # 5 minutes
        echo "⏳ Building wheels... $(date '+%H:%M:%S') - Keep Colab alive ping"
        echo "📊 Current wheel cache size: $(du -sh $WHEEL_CACHE_DIR 2>/dev/null | cut -f1 || echo '0B')"
        # Show some system activity to prevent timeout
        df -h /tmp | tail -1
    done
}

# Start keep-alive background process
echo "🔄 Starting keep-alive process to prevent Colab timeout..."
keep_alive &
KEEP_ALIVE_PID=$!

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up..."
    kill $KEEP_ALIVE_PID 2>/dev/null || true
    rm -rf "$TEMP_WHEEL_DIR"
}
trap cleanup EXIT

echo "🚀 Starting wheel building process..."
echo "📝 Building wheels for $(wc -l < $REQUIREMENTS_FILE) packages..."
echo ""

# Set pip configuration for wheel building
export PIP_CACHE_DIR="/tmp/pip_cache"
export PIP_WHEEL_DIR="$TEMP_WHEEL_DIR"
mkdir -p "$PIP_CACHE_DIR"

# CRITICAL FIX: Install PyTorch first (foundational dependency) - CUDA 12.1
# Version 2.7.1 to match install_from_wheels.sh requirements
echo "🔥 Building PyTorch 2.7.1 ecosystem wheels..."
python3 -m pip wheel \
    "torch==2.7.1" \
    "torchvision==0.22.1" \
    "torchaudio==2.7.1" \
    --wheel-dir="$TEMP_WHEEL_DIR" \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --prefer-binary

echo "✅ PyTorch wheels built"

# CRITICAL FIX: Install core dependencies in order
echo "📦 Installing core dependencies..."
python3 -m pip wheel \
    "setuptools" \
    "wheel" \
    "cython" \
    "packaging" \
    --wheel-dir="$TEMP_WHEEL_DIR" \
    --prefer-binary

# Download direct wheel URLs first (mamba-ssm and causal-conv1d)
echo "🔧 Downloading special packages with direct GitHub URLs..."

echo "⬇️ Downloading causal-conv1d wheel (PyTorch 2.7 compatible)..."
wget -P "$TEMP_WHEEL_DIR" https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl || echo "⚠️ causal-conv1d download failed"

echo "⬇️ Downloading mamba-ssm wheel (PyTorch 2.7 compatible)..."
wget -P "$TEMP_WHEEL_DIR" https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl || echo "⚠️ mamba-ssm download failed"

echo "✅ Special packages wheels downloaded"
echo ""

# OPTIMIZATION: Pre-cache GitHub direct URL wheels (556.9MB total)
echo "⬇️ Pre-caching large GitHub direct URL wheels..."
echo "🔥 This will save 556.9MB of downloads on every install!"
echo ""

# Download mamba-ssm wheel directly (423.9 MB)
if [ ! -f "$WHEEL_CACHE_DIR/mamba_ssm-2.2.5+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" ]; then
    echo "📥 Downloading mamba-ssm wheel (423.9MB)..."
    wget -q --show-progress -O "$WHEEL_CACHE_DIR/mamba_ssm-2.2.5+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
        "https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    echo "✅ mamba-ssm wheel cached"
else
    echo "✅ mamba-ssm wheel already cached"
fi

# Download causal-conv1d wheel directly (133.0 MB) 
if [ ! -f "$WHEEL_CACHE_DIR/causal_conv1d-1.5.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" ]; then
    echo "📥 Downloading causal-conv1d wheel (133.0MB)..."
    wget -q --show-progress -O "$WHEEL_CACHE_DIR/causal_conv1d-1.5.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    echo "✅ causal-conv1d wheel cached"
else
    echo "✅ causal-conv1d wheel already cached"
fi

echo ""

# CRITICAL FIX: Build wheels with proper error handling and retries
echo "🏗️ Building wheels for all requirements with error handling..."
echo "⚠️ Some packages may fail to build wheels - this is normal"
echo "📊 Progress will be shown every 5 minutes"
echo ""

# OPTIMIZATION: Pre-cache heaviest PyPI packages first (200+ MB total)
echo "📦 Pre-building wheels for heaviest packages..."
HEAVY_PACKAGES=(
    "scipy==1.15.3"           # 37.7 MB
    "scikit-learn==1.5.2"    # 13.3 MB - Required by TimesFM
    "pandas==2.3.2"          # 12.3 MB
    "plotly==6.3.0"          # 9.8 MB
    "transformers==4.44.0"   # 9.5 MB - Required by mamba_ssm
    "matplotlib==3.10.5"     # 8.7 MB
    "timesfm==1.3.0"         # TimesFM main package
    "wandb>=0.17.5"          # Required by TimesFM
    "huggingface-hub>=0.23.0" # Required by TimesFM
    "ninja==1.11.1.1"       # Required by mamba_ssm and causal-conv1d
    "einops==0.8.0"         # Required by mamba_ssm
    "accelerate==0.34.0"    # ML acceleration
    "absl-py>=1.4.0"        # TimesFM dependencies
    "einshape>=1.0.0"        
    "typer>=0.12.3"
    "utilsforecast>=0.1.10"
)

for package in "${HEAVY_PACKAGES[@]}"; do
    echo "🔧 Pre-building wheel for $package..."
    python3 -m pip wheel "$package" --wheel-dir="$TEMP_WHEEL_DIR" \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --prefer-binary \
        --find-links "$TEMP_WHEEL_DIR" \
        --no-build-isolation || echo "⚠️ Failed to pre-build $package - will try in main build"
done

echo "✅ Heavy packages pre-cached"
echo ""

# Use pip wheel with --no-build-isolation to fix setup.py issues
# Updated PyG index to match PyTorch 2.7
python3 -m pip wheel -r "$REQUIREMENTS_FILE" --wheel-dir="$TEMP_WHEEL_DIR" \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://data.pyg.org/whl/torch-2.7.0+cu121.html \
    --prefer-binary \
    --find-links "$TEMP_WHEEL_DIR" \
    --no-build-isolation || echo "⚠️ Some wheels failed to build - continuing with available wheels"

echo ""
echo "✅ Wheel building process completed!"

# Move wheels to persistent cache
echo "💾 Moving wheels to persistent cache..."
rsync -av "$TEMP_WHEEL_DIR/" "$WHEEL_CACHE_DIR/"

# Create wheel index
echo "📋 Creating wheel index..."
ls -la "$WHEEL_CACHE_DIR"/*.whl > "$WHEEL_CACHE_DIR/wheel_index.txt" 2>/dev/null || true

# Show final statistics
WHEEL_COUNT=$(ls -1 "$WHEEL_CACHE_DIR"/*.whl 2>/dev/null | wc -l || echo "0")
CACHE_SIZE=$(du -sh "$WHEEL_CACHE_DIR" 2>/dev/null | cut -f1 || echo "0B")

echo ""
echo "🎉 WHEEL BUILDING COMPLETE!"
echo "================================"
echo "📦 Built wheels: $WHEEL_COUNT"
echo "💾 Cache size: $CACHE_SIZE"
echo "📂 Cache location: $WHEEL_CACHE_DIR"
echo ""
echo "⚡ Next time, use install_from_wheels.sh for 2-3 minute installation!"
echo "💡 Wheels are saved to Google Drive and will persist across sessions"
echo ""

# Verify a few key packages
echo "🔍 Verifying key wheel files..."
for pkg in torch numpy pandas transformers; do
    if ls "$WHEEL_CACHE_DIR"/${pkg}*.whl >/dev/null 2>&1; then
        echo "✅ $pkg wheel found"
    else
        echo "⚠️ $pkg wheel missing"
    fi
done

echo ""
echo "✅ Wheel cache is ready for fast installations!"