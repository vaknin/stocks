#!/bin/bash
# Fast installation from cached wheels - 2-3 minutes
# Use this after running build_wheels.sh once

set -e  # Exit on any error

echo "⚡ FAST INSTALL FROM CACHED WHEELS"
echo "=================================="
echo "🚀 Expected time: 2-3 minutes"
echo ""

# Configuration
WHEEL_CACHE_DIR="/content/drive/MyDrive/colab_wheels"
REQUIREMENTS_FILE="/content/colab_setup/colab_requirements.txt"

# Check if wheel cache exists
if [ ! -d "$WHEEL_CACHE_DIR" ]; then
    echo "❌ Wheel cache not found at: $WHEEL_CACHE_DIR"
    echo "💡 Run build_wheels.sh first to create the wheel cache"
    exit 1
fi

# Check if wheels exist
WHEEL_COUNT=$(ls -1 "$WHEEL_CACHE_DIR"/*.whl 2>/dev/null | wc -l || echo "0")
if [ "$WHEEL_COUNT" -eq "0" ]; then
    echo "❌ No wheels found in cache directory: $WHEEL_CACHE_DIR"
    echo "💡 Run build_wheels.sh first to build the wheel cache"
    exit 1
fi

echo "✅ Found wheel cache with $WHEEL_COUNT wheels"
echo "📂 Cache location: $WHEEL_CACHE_DIR"
echo ""

# Verify requirements file
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "❌ Requirements file not found: $REQUIREMENTS_FILE"
    exit 1
fi

echo "✅ Requirements file found: $REQUIREMENTS_FILE"
echo ""

# Install Python 3.10 if needed
if ! command -v python3.10 &> /dev/null; then
    echo "🐍 Installing Python 3.10..."
    sudo apt-get update -y
    sudo apt-get install python3.10 python3.10-distutils -y
    
    # Set up alternatives
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 2>/dev/null || true
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2
    sudo update-alternatives --set python3 /usr/bin/python3.10
    
    # Install pip for Python 3.10
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
    echo "✅ Python 3.10 installed"
else
    echo "✅ Python 3.10 already available"
fi

echo ""
echo "🚀 Installing packages from cached wheels and direct URLs..."
echo "⏱️ This should take 2-3 minutes..."

# Install direct URL packages first (mamba-ssm and causal-conv1d)
echo "⬇️ Installing special packages from direct URLs..."
python3 -m pip install --no-cache-dir \
    https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install remaining packages from cached wheels with fallback to PyPI
echo "📦 Installing remaining packages from cached wheels..."
# Create temporary requirements file without the direct URL packages
grep -v "^https://github.com/" "$REQUIREMENTS_FILE" > /tmp/requirements_filtered.txt

python3 -m pip install --find-links "$WHEEL_CACHE_DIR" \
    --prefer-binary --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu125 \
    -r /tmp/requirements_filtered.txt

echo ""
echo "✅ INSTALLATION COMPLETE!"
echo "========================"

# Verify key installations
echo "🔍 Verifying installations..."

python3 -c "
import sys
print(f'Python version: {sys.version}')
print()

packages_to_check = [
    'torch', 'numpy', 'pandas', 'transformers', 
    'mamba_ssm', 'causal_conv1d', 'yfinance', 
    'loguru', 'pydantic'
]

success_count = 0
for package in packages_to_check:
    try:
        __import__(package)
        print(f'✅ {package}')
        success_count += 1
    except ImportError as e:
        print(f'❌ {package}: {e}')

print()
print(f'✅ Successfully verified {success_count}/{len(packages_to_check)} packages')

if success_count == len(packages_to_check):
    print('🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!')
    print('🚀 Your trading system is ready to use!')
else:
    print('⚠️ Some packages failed - check the output above')
"

echo ""
echo "📊 Installation Statistics:"
echo "- Wheel cache used: $WHEEL_CACHE_DIR"
echo "- Wheels available: $WHEEL_COUNT"
echo "- Installation time: ~2-3 minutes (vs 30+ minutes fresh install)"
echo ""
echo "💡 To rebuild the wheel cache, run: bash build_wheels.sh"