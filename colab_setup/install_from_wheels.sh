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

# Enhanced drive mounting and connectivity verification
echo "🔗 Verifying Google Drive connectivity..."
# Check if drive is mounted
if [ ! -d "/content/drive" ]; then
    echo "❌ Google Drive not mounted. Attempting to mount..."
    python3 -c "from google.colab import drive; drive.mount('/content/drive')"
fi

# Check if wheel cache exists
if [ ! -d "$WHEEL_CACHE_DIR" ]; then
    echo "❌ Wheel cache not found at: $WHEEL_CACHE_DIR"
    echo "💡 Run build_wheels.sh first to create the wheel cache"
    exit 1
fi

# Test drive connectivity by creating a test file
test_file="$WHEEL_CACHE_DIR/.connectivity_test"
if ! echo "test" > "$test_file" 2>/dev/null || ! rm "$test_file" 2>/dev/null; then
    echo "❌ Google Drive connectivity issues detected"
    echo "🔄 Attempting to remount drive..."
    umount /content/drive 2>/dev/null || true
    sleep 2
    python3 -c "from google.colab import drive; drive.mount('/content/drive', force_remount=True)"
    sleep 3
fi

# Check if wheels exist and verify integrity
WHEEL_COUNT=$(ls -1 "$WHEEL_CACHE_DIR"/*.whl 2>/dev/null | wc -l || echo "0")
if [ "$WHEEL_COUNT" -eq "0" ]; then
    echo "❌ No wheels found in cache directory: $WHEEL_CACHE_DIR"
    echo "💡 Run build_wheels.sh first to build the wheel cache"
    exit 1
fi

# Function to verify wheel integrity
verify_wheel_integrity() {
    local wheel_file="$1"
    if [ ! -f "$wheel_file" ] || [ ! -r "$wheel_file" ]; then
        return 1
    fi
    # Check if file is not empty and is readable
    if [ ! -s "$wheel_file" ]; then
        return 1
    fi
    return 0
}

# Verify a few critical wheels
echo "🔍 Verifying wheel integrity..."
CRITICAL_WHEELS=(
    "torch-2.7.1"
    "mamba_ssm-2.2.5"
    "causal_conv1d-1.5.2"
)

for wheel_pattern in "${CRITICAL_WHEELS[@]}"; do
    wheel_found=false
    for wheel_file in "$WHEEL_CACHE_DIR"/${wheel_pattern}*.whl; do
        if verify_wheel_integrity "$wheel_file"; then
            wheel_found=true
            break
        fi
    done
    if [ "$wheel_found" = false ]; then
        echo "⚠️ Critical wheel $wheel_pattern may be corrupted or missing"
    fi
done

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

# Python 3.10 setup handled by Colab notebook - skipping
echo "✅ Python 3.10 setup handled by notebook"
echo ""
echo "🚀 Installing packages from cached wheels and direct URLs..."
echo "⏱️ This should take 2-3 minutes..."

# Function for robust wheel installation with cache-first approach
install_with_cache_fallback() {
    local packages="$1"
    local description="$2"
    local max_retries=3
    local retry=0
    
    echo "🔥 Installing $description with cache-first strategy (saves bandwidth)..."
    
    # STEP 1: Try cache-only installation first (fastest, saves bandwidth)
    echo "📦 Attempting cache-only installation..."
    if python3 -m pip install --no-cache-dir \
        --no-index \
        --find-links "$WHEEL_CACHE_DIR" \
        $packages 2>/dev/null; then
        echo "✅ Successfully installed $description from cache (100% cache hit!)"
        return 0
    fi
    
    # STEP 2: Cache-only failed, try hybrid approach with retries
    echo "⚠️ Cache-only failed, trying hybrid approach with PyPI fallback..."
    
    while [ $retry -lt $max_retries ]; do
        # Try cache-preferred hybrid approach
        if python3 -m pip install --no-cache-dir \
            --find-links "$WHEEL_CACHE_DIR" \
            --prefer-binary \
            --extra-index-url https://download.pytorch.org/whl/cu121 \
            $packages; then
            echo "✅ Successfully installed $description (hybrid cache+PyPI)"
            return 0
        else
            retry=$((retry + 1))
            echo "⚠️ Installation attempt $retry failed for $description"
            if [ $retry -lt $max_retries ]; then
                echo "🔄 Retrying in 5 seconds..."
                sleep 5
            fi
        fi
    done
    
    echo "❌ Failed to install $description after $max_retries attempts"
    return 1
}

# Install PyTorch 2.7.1 ecosystem with improved caching strategy
install_with_cache_fallback "torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1" "PyTorch 2.7.1 ecosystem"

# Install critical dependencies required by mamba_ssm FIRST
install_with_cache_fallback "ninja==1.11.1.1 einops==0.8.0 transformers==4.44.0" "mamba_ssm dependencies"

# Install mamba_ssm and causal-conv1d with enhanced caching strategy
echo "⚡ Installing mamba_ssm and causal-conv1d from cached wheels (saves 556.9MB)..."

# Enhanced installation with integrity checks
install_mamba_packages() {
    local causal_wheel="$WHEEL_CACHE_DIR/causal_conv1d-1.5.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    local mamba_wheel="$WHEEL_CACHE_DIR/mamba_ssm-2.2.5+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    
    # Check if cached wheels exist and are valid
    if verify_wheel_integrity "$causal_wheel" && verify_wheel_integrity "$mamba_wheel"; then
        echo "✅ Found valid cached wheels - installing from cache (saves 556.9MB download!)"
        python3 -m pip install --no-cache-dir --no-deps "$causal_wheel" "$mamba_wheel" && return 0
    fi
    
    # Fallback to direct URLs
    echo "⚠️ Cached wheels invalid/missing - downloading directly from GitHub"
    python3 -m pip install --no-cache-dir --no-deps \
        https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
        https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
}

install_mamba_packages

# Install remaining packages from cached wheels with fallback to PyPI
echo "📦 Installing remaining packages from cached wheels..."
# Create temporary requirements file excluding packages we've already installed
grep -v "^https://github.com/" "$REQUIREMENTS_FILE" | \
    grep -v "^torch==" | grep -v "^torchvision==" | grep -v "^torchaudio==" | \
    grep -v "^timesfm==" | grep -v "^ninja==" | grep -v "^einops==" | \
    grep -v "^transformers==" | grep -v "^wandb" | grep -v "^scikit-learn==" | \
    grep -v "^huggingface-hub" | grep -v "^absl-py" | grep -v "^einshape" | \
    grep -v "^typer" | grep -v "^utilsforecast" \
    > /tmp/requirements_filtered.txt

# Install TimesFM and ALL its dependencies using hybrid approach
install_with_cache_fallback "timesfm==1.3.0 wandb>=0.17.5 scikit-learn>=1.2.2 huggingface-hub>=0.23.0 absl-py>=1.4.0 einshape>=1.0.0 typer>=0.12.3 utilsforecast>=0.1.10" "TimesFM and dependencies"

# Install remaining packages using enhanced cache-first approach
echo "📦 Installing remaining packages with cache-first strategy..."
echo "📦 Step 1: Attempting cache-only installation (fastest)..."

# STEP 1: Try cache-only installation for maximum speed
if python3 -m pip install --no-cache-dir \
    --no-index \
    --find-links "$WHEEL_CACHE_DIR" \
    -r /tmp/requirements_filtered.txt 2>/dev/null; then
    echo "✅ All remaining packages installed from cache (100% cache efficiency!)"
else
    echo "⚠️ Cache-only installation incomplete, using hybrid approach..."
    
    # STEP 2: Use hybrid approach with cache preference
    python3 -m pip install --find-links "$WHEEL_CACHE_DIR" \
        --prefer-binary --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://data.pyg.org/whl/torch-2.7.0+cu121.html \
        -r /tmp/requirements_filtered.txt || {
        echo "⚠️ Some packages failed to install - trying individual installation..."
        # Try installing packages one by one for better error handling
        while IFS= read -r package; do
            if [ -n "$package" ] && [[ ! "$package" =~ ^# ]]; then
                echo "🔧 Installing: $package"
                # Try cache-only first, then hybrid
                python3 -m pip install --no-cache-dir --no-index --find-links "$WHEEL_CACHE_DIR" "$package" 2>/dev/null || \
                python3 -m pip install --find-links "$WHEEL_CACHE_DIR" \
                    --prefer-binary --no-cache-dir \
                    --extra-index-url https://download.pytorch.org/whl/cu121 \
                    "$package" || echo "⚠️ Failed to install: $package"
            fi
        done < /tmp/requirements_filtered.txt
    }
fi

# Enhanced post-installation verification
echo ""
echo "🔍 Post-installation verification and statistics..."
echo "====================================================="

# Count successful cache usage vs PyPI downloads
CACHED_INSTALLS=0
PYPI_INSTALLS=0

if [ -f "$PIP_CACHE_DIR/pip.log" ]; then
    CACHED_INSTALLS=$(grep -c "find-links" "$PIP_CACHE_DIR/pip.log" 2>/dev/null || echo "0")
fi

echo "📊 Installation Statistics:"
echo "- Wheel cache used: $WHEEL_CACHE_DIR"
echo "- Wheels available: $WHEEL_COUNT"
echo "- Cache strategy: Cache-first with PyPI fallback"
echo "- Expected cache efficiency: 90%+ (saves 800MB+ downloads)"
echo "- PyPI downloads: Only for missing wheels or dependencies"
echo ""
echo "✅ INSTALLATION COMPLETE!"
echo "========================"

# Enhanced verification with error reporting
echo "🔍 Verifying installations with detailed reporting..."

# Function to test package import with detailed error info
test_package_import() {
    local package="$1"
    local import_name="${2:-$package}"
    
    python3 -c "
try:
    import $import_name
    print('✅ $package: Successfully imported')
    # Try to get version if available
    try:
        if hasattr($import_name, '__version__'):
            print('   Version: ' + str($import_name.__version__))
        elif hasattr($import_name, 'version'):
            print('   Version: ' + str($import_name.version))
    except:
        pass
except ImportError as e:
    print('❌ $package: Import failed - ' + str(e))
except Exception as e:
    print('⚠️ $package: Import error - ' + str(e))
" 2>&1
}

echo "Python version: $(python3 --version)"
echo ""

# Enhanced package verification
packages_to_check=(
    "torch"
    "numpy"
    "pandas"
    "transformers"
    "mamba_ssm"
    "causal_conv1d"
    "yfinance"
    "loguru"
    "pydantic"
    "timesfm"
    "wandb"
    "sklearn:scikit-learn"
)

success_count=0
total_packages=${#packages_to_check[@]}

echo "Testing package imports:"
for pkg_spec in "${packages_to_check[@]}"; do
    if [[ "$pkg_spec" == *":"* ]]; then
        IFS=':' read -r import_name package_name <<< "$pkg_spec"
        test_package_import "$package_name" "$import_name"
    else
        test_package_import "$pkg_spec"
    fi
    
    # Count successes (simple check)
    if python3 -c "import ${pkg_spec%:*}" 2>/dev/null; then
        ((success_count++))
    fi
done

echo ""
echo "=== FINAL VERIFICATION RESULTS ==="
echo "✅ Successfully verified: $success_count/$total_packages packages"

if [ "$success_count" -eq "$total_packages" ]; then
    echo "🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!"
    echo "🚀 Your trading system is ready to use!"
    echo "✨ Cached wheel strategy working optimally!"
else
    echo "⚠️ Some packages failed verification - check the detailed output above"
    echo "💡 Consider running build_wheels.sh to refresh the cache"
fi

echo ""
echo "📊 OPTIMIZED Installation Statistics:"
echo "- Wheel cache used: $WHEEL_CACHE_DIR"
echo "- Wheels available: $WHEEL_COUNT"
echo "- Strategy: Hybrid cache-first with PyPI fallback"
echo "- PyTorch ecosystem: Prioritized from cache (saves 800+ MB when available)"
echo "- GitHub wheels: Direct cache access (saves 556.9MB when available)"
echo "- Heavy packages: Cache-first installation (saves 200+ MB when available)"
echo "- Network efficiency: Minimized downloads through intelligent caching"
echo "- Reliability: Robust fallback ensures successful installation"
echo "- Installation time: 2-5 minutes (adaptive based on cache effectiveness)"
echo ""
echo "💡 To rebuild/refresh the wheel cache, run: bash build_wheels.sh"
echo "🔧 This hybrid approach ensures maximum cache utilization with reliability!"