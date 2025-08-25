#!/bin/bash
# Google Colab AI Trading System Installer with Wheel Caching
# Smart installer that chooses the best installation method

set -e  # Exit on any error

echo "🤖 AI TRADING SYSTEM - GOOGLE COLAB INSTALLER"
echo "============================================="
echo "🚀 Smart installer with multiple installation modes"
echo ""

# Configuration
WHEEL_CACHE_DIR="/content/drive/MyDrive/colab_wheels"
REQUIREMENTS_FILE="/content/colab_setup/colab_requirements.txt"

# Installation mode selection
show_menu() {
    echo "📋 INSTALLATION OPTIONS:"
    echo "========================"
    echo "1. ⚡ Fast Install (2-3 min) - Use cached wheels"
    echo "2. 🏗️ Build Wheels Cache (60-120 min) - One-time setup for fast future installs"
    echo "3. 📦 Chunked Install (20-40 min) - Timeout-resistant fresh install"
    echo "4. 🔍 Check Cache Status - See if wheels are available"
    echo "5. 🧪 Verify Installation - Test if packages work"
    echo ""
}

check_cache_status() {
    echo "🔍 CHECKING WHEEL CACHE STATUS"
    echo "=============================="
    
    if [ -d "$WHEEL_CACHE_DIR" ]; then
        WHEEL_COUNT=$(ls -1 "$WHEEL_CACHE_DIR"/*.whl 2>/dev/null | wc -l || echo "0")
        CACHE_SIZE=$(du -sh "$WHEEL_CACHE_DIR" 2>/dev/null | cut -f1 || echo "0B")
        
        if [ "$WHEEL_COUNT" -gt "0" ]; then
            echo "✅ Wheel cache found!"
            echo "📂 Location: $WHEEL_CACHE_DIR"
            echo "📦 Wheels available: $WHEEL_COUNT"
            echo "💾 Cache size: $CACHE_SIZE"
            echo ""
            
            # Check for key packages
            echo "🔍 Key packages in cache:"
            for pkg in torch numpy pandas transformers mamba_ssm; do
                if ls "$WHEEL_CACHE_DIR"/${pkg}*.whl >/dev/null 2>&1; then
                    echo "✅ $pkg"
                else
                    echo "❌ $pkg"
                fi
            done
            echo ""
            echo "💡 You can use Option 1 (Fast Install) for 2-3 minute installation!"
            return 0
        else
            echo "⚠️ Cache directory exists but no wheels found"
            echo "💡 Use Option 2 (Build Wheels Cache) to create cache"
            return 1
        fi
    else
        echo "❌ No wheel cache found"
        echo "💡 Use Option 2 (Build Wheels Cache) to create cache for future fast installs"
        return 1
    fi
}

verify_installation() {
    echo "🧪 VERIFYING INSTALLATION"
    echo "========================="
    
    python3 -c "
import sys
print(f'Python version: {sys.version}')
print()

# Test critical packages
test_results = []

print('Testing package imports...')
packages_to_test = [
    ('torch', 'PyTorch (Deep Learning)'),
    ('numpy', 'NumPy (Numerical Computing)'),
    ('pandas', 'Pandas (Data Analysis)'),
    ('transformers', 'Transformers (AI Models)'),
    ('yfinance', 'yFinance (Market Data)'),
    ('loguru', 'Loguru (Logging)'),
    ('pydantic', 'Pydantic (Data Validation)'),
    ('sklearn', 'Scikit-learn (Machine Learning)'),
]

success_count = 0
for package_name, description in packages_to_test:
    try:
        __import__(package_name)
        print(f'✅ {description}')
        success_count += 1
    except ImportError as e:
        print(f'❌ {description}: {e}')

print()

# Test advanced packages
print('Testing advanced AI packages...')
advanced_packages = [
    ('mamba_ssm', 'Mamba State Space Models'),
    ('causal_conv1d', 'Causal Conv1D'),
]

advanced_success = 0
for package_name, description in advanced_packages:
    try:
        __import__(package_name)
        print(f'✅ {description}')
        advanced_success += 1
    except ImportError as e:
        print(f'❌ {description}: {e}')

print()
print(f'📊 SUMMARY:')
print(f'✅ Basic packages: {success_count}/{len(packages_to_test)}')
print(f'✅ Advanced packages: {advanced_success}/{len(advanced_packages)}')

total_success = success_count + advanced_success
total_packages = len(packages_to_test) + len(advanced_packages)

if total_success == total_packages:
    print()
    print('🎉 ALL PACKAGES WORKING PERFECTLY!')
    print('🚀 Your AI trading system is ready to use!')
    print('💡 You can now run your trading scripts')
elif success_count == len(packages_to_test):
    print()
    print('✅ CORE PACKAGES WORKING!')
    print('⚠️  Some advanced AI packages missing, but basic functionality available')
    print('🚀 Your trading system should work for basic operations')
else:
    print()
    print('❌ INSTALLATION INCOMPLETE')
    print('🔧 Try reinstalling or check the logs for errors')
"
}

auto_detect_best_method() {
    echo "🤖 AUTO-DETECTING BEST INSTALLATION METHOD"
    echo "=========================================="
    
    if check_cache_status >/dev/null 2>&1; then
        echo "✅ Wheel cache detected - Using fast install method"
        return 1  # Fast install
    else
        echo "❌ No wheel cache found"
        echo "💡 For first-time setup, recommend building wheel cache for future speed"
        echo "🔄 Falling back to chunked install for now"
        return 3  # Chunked install
    fi
}

# Parse command line arguments
if [ $# -gt 0 ]; then
    case "$1" in
        "fast"|"1")
            CHOICE=1
            ;;
        "build"|"2")
            CHOICE=2
            ;;
        "chunked"|"3")
            CHOICE=3
            ;;
        "check"|"4")
            CHOICE=4
            ;;
        "verify"|"5")
            CHOICE=5
            ;;
        "auto")
            auto_detect_best_method
            CHOICE=$?
            ;;
        *)
            echo "❌ Invalid option: $1"
            echo "Valid options: fast, build, chunked, check, verify, auto"
            exit 1
            ;;
    esac
else
    # Interactive mode
    show_menu
    echo -n "Choose an option (1-5) or 'auto' for automatic detection: "
    read CHOICE
    
    if [ "$CHOICE" = "auto" ]; then
        auto_detect_best_method
        CHOICE=$?
    fi
fi

echo ""

# Execute chosen option
case $CHOICE in
    1)
        echo "⚡ FAST INSTALL SELECTED"
        echo "======================="
        if [ -f "./install_from_wheels.sh" ]; then
            bash ./install_from_wheels.sh
        else
            echo "❌ install_from_wheels.sh not found"
            exit 1
        fi
        ;;
    2)
        echo "🏗️ BUILD WHEELS CACHE SELECTED"
        echo "=============================="
        echo "⚠️ This will take 60-120 minutes but only needs to be done once"
        echo "✅ Future installs will be 2-3 minutes"
        echo ""
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if [ -f "./build_wheels.sh" ]; then
                bash ./build_wheels.sh
            else
                echo "❌ build_wheels.sh not found"
                exit 1
            fi
        else
            echo "❌ Cancelled by user"
            exit 1
        fi
        ;;
    3)
        echo "📦 CHUNKED INSTALL SELECTED"
        echo "=========================="
        if [ -f "./install_colab_chunked.sh" ]; then
            bash ./install_colab_chunked.sh
        else
            echo "❌ install_colab_chunked.sh not found"
            exit 1
        fi
        ;;
    4)
        check_cache_status
        ;;
    5)
        verify_installation
        ;;
    *)
        echo "❌ Invalid choice: $CHOICE"
        show_menu
        exit 1
        ;;
esac

echo ""
echo "🎯 NEXT STEPS:"
echo "============="
echo "1. 🧪 Run verification: bash install_colab.sh verify"
echo "2. 📊 Test your trading system with sample data"
echo "3. 💡 For fastest future installs, build wheel cache once (Option 2)"
echo ""
echo "📚 Installation methods summary:"
echo "- Fast Install: 2-3 min (requires wheel cache)"
echo "- Build Cache: 60-120 min (one-time setup)"
echo "- Chunked Install: 20-40 min (no cache needed)"