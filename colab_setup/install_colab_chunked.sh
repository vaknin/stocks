#!/bin/bash
# Chunked installation to avoid 40-minute timeout
# Installs packages in smaller groups with progress monitoring

set -e  # Exit on any error

echo "📦 CHUNKED INSTALLATION FOR GOOGLE COLAB"
echo "========================================"
echo "🎯 Installs packages in chunks to avoid timeout"
echo "⏱️ Expected time: 20-40 minutes (depending on connection)"
echo ""

# Configuration
REQUIREMENTS_FILE="/content/colab_setup/colab_requirements.txt"
CHUNK_SIZE=8  # Install 8 packages at a time
TIMEOUT_PER_CHUNK=900  # 15 minutes per chunk

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

# Function to keep Colab alive
keep_alive() {
    while true; do
        sleep 180  # 3 minutes
        echo "⏳ Installing packages... $(date '+%H:%M:%S') - Keep Colab alive"
        # Show system info to prevent timeout
        free -h | head -2
    done
}

# Start keep-alive background process
keep_alive &
KEEP_ALIVE_PID=$!

# Cleanup function
cleanup() {
    echo "🧹 Stopping keep-alive process..."
    kill $KEEP_ALIVE_PID 2>/dev/null || true
}
trap cleanup EXIT

# Install critical packages first (these are most likely to timeout)
echo "🔧 Installing critical packages first..."

# PyTorch ecosystem - CUDA 12.1 (backward compatible with 12.5)
echo "⚡ Chunk 0: PyTorch ecosystem (most critical)"
timeout $TIMEOUT_PER_CHUNK python3 -m pip install \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --prefer-binary --no-cache-dir

echo "✅ PyTorch installed successfully"

# Mamba and causal-conv1d (direct GitHub wheels to avoid build issues)
echo "⚡ Chunk 1: Mamba ecosystem (AI models)"
timeout $TIMEOUT_PER_CHUNK python3 -m pip install \
    https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    --no-cache-dir

echo "✅ Mamba ecosystem installed successfully"

# Transformers and acceleration
echo "⚡ Chunk 2: Transformers and acceleration"
timeout $TIMEOUT_PER_CHUNK python3 -m pip install \
    transformers==4.44.0 accelerate==0.34.0 einops==0.8.0 \
    --prefer-binary --no-cache-dir

echo "✅ Transformers installed successfully"

# Create chunks for remaining packages
echo "📝 Creating chunks for remaining packages..."

# Read requirements and filter out already installed packages and direct URLs
TEMP_REQUIREMENTS="/tmp/remaining_requirements.txt"
grep -v "^torch==" "$REQUIREMENTS_FILE" | \
grep -v "^torchvision==" | \
grep -v "^torchaudio==" | \
grep -v "^https://github.com/" | \
grep -v "^transformers==" | \
grep -v "^accelerate==" | \
grep -v "^einops==" | \
grep -v "^#" | \
grep -v "^$" > "$TEMP_REQUIREMENTS"

# Count remaining packages
REMAINING_COUNT=$(wc -l < "$TEMP_REQUIREMENTS")
TOTAL_CHUNKS=$(((REMAINING_COUNT + CHUNK_SIZE - 1) / CHUNK_SIZE))

echo "📊 Remaining packages: $REMAINING_COUNT"
echo "📦 Will install in $TOTAL_CHUNKS chunks of $CHUNK_SIZE packages each"
echo ""

# Install remaining packages in chunks
chunk_num=3
while IFS= read -r line; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    
    # Collect packages for this chunk
    packages=()
    packages+=("$line")
    
    # Read more packages for this chunk
    for ((i=1; i<CHUNK_SIZE; i++)); do
        if IFS= read -r next_line; then
            [[ -z "$next_line" || "$next_line" =~ ^# ]] && continue
            packages+=("$next_line")
        else
            break
        fi
    done
    
    # Install this chunk
    if [ ${#packages[@]} -gt 0 ]; then
        echo "⚡ Chunk $chunk_num: Installing ${#packages[@]} packages"
        echo "📦 Packages: ${packages[*]}"
        
        timeout $TIMEOUT_PER_CHUNK python3 -m pip install "${packages[@]}" \
            --extra-index-url https://download.pytorch.org/whl/cu121 \
            --extra-index-url https://data.pyg.org/whl/torch-2.4.0+cu121.html \
            --prefer-binary --no-cache-dir
        
        echo "✅ Chunk $chunk_num completed successfully"
        echo ""
        
        ((chunk_num++))
    fi
    
done < "$TEMP_REQUIREMENTS"

echo "🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!"
echo "======================================"

# Verify installation
echo "🔍 Verifying critical packages..."
python3 -c "
import sys
print(f'Python version: {sys.version}')
print()

packages_to_check = [
    'torch', 'numpy', 'pandas', 'transformers', 
    'mamba_ssm', 'causal_conv1d', 'yfinance', 
    'loguru', 'pydantic', 'scikit_learn'
]

success_count = 0
for package in packages_to_check:
    try:
        if package == 'scikit_learn':
            import sklearn
            package = 'sklearn'
        else:
            __import__(package)
        print(f'✅ {package}')
        success_count += 1
    except ImportError as e:
        print(f'❌ {package}: {e}')

print()
print(f'✅ Successfully verified {success_count}/{len(packages_to_check)} packages')

if success_count == len(packages_to_check):
    print('🎉 ALL CRITICAL PACKAGES WORKING!')
    print('🚀 Your trading system is ready to use!')
else:
    print('⚠️ Some packages failed - but installation may still be usable')
"

echo ""
echo "📊 Installation Statistics:"
echo "- Installation method: Chunked (timeout-resistant)"
echo "- Chunk size: $CHUNK_SIZE packages"
echo "- Timeout per chunk: $((TIMEOUT_PER_CHUNK/60)) minutes"
echo "- Total chunks processed: $((chunk_num-1))"
echo ""
echo "💡 For even faster future installs, consider using build_wheels.sh + install_from_wheels.sh"

# Cleanup
rm -f "$TEMP_REQUIREMENTS"