#!/bin/bash

# Optimized Python 3.10 Setup - Fully Automatic (30-60 seconds)
echo "🐍 AUTOMATED PYTHON 3.10 SETUP"
echo "==============================="
echo "⏱️ Expected time: 30-60 seconds"
echo ""

# Check current Python version
CURRENT_PYTHON=$(python3 --version 2>/dev/null || echo "None")
echo "📋 Current Python: $CURRENT_PYTHON"

# Update package list (suppress harmless r2u warning)
echo "📦 Updating package list..."
sudo apt-get update -y 2>/dev/null || sudo apt-get update -y

# Install Python 3.10 if needed
echo "🔧 Ensuring Python 3.10 is available..."
sudo apt-get install python3.10 python3-distutils -y

# Configure Python 3.10 as default (NON-INTERACTIVE)
echo "⚙️ Setting Python 3.10 as default..."
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
sudo update-alternatives --set python3 /usr/bin/python3.10

# Verify Python version switch
NEW_PYTHON=$(python3 --version 2>/dev/null || echo "Failed")
echo "✅ New Python version: $NEW_PYTHON"

# Install/upgrade pip for Python 3.10 if needed
if ! python3.10 -m pip --version >/dev/null 2>&1; then
    echo "🔧 Installing pip for Python 3.10..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
else
    echo "✅ pip already available for Python 3.10"
fi

# Final verification
echo ""
echo "✅ PYTHON SETUP COMPLETE!"
echo "========================"
python3 -c "
import sys
print(f'🐍 Python: {sys.version.split()[0]}')
print(f'📍 Path: {sys.executable}')
print(f'🔧 Pip: ', end='')
try:
    import pip
    print(pip.__version__)
except:
    print('Not available')
"