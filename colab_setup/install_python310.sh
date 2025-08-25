#!/bin/bash
# Install Python 3.10 in Google Colab
# Python 3.10 has better package compatibility than 3.11

set -e  # Exit on any error

echo "🐍 Installing Python 3.10 in Google Colab..."

# Update package list
echo "📦 Updating package list..."
sudo apt-get update -y

# Install Python 3.10 and distutils
echo "⬇️ Installing Python 3.10..."
sudo apt-get install python3.10 python3.10-distutils -y

# Set up alternatives (Python 3.9 priority 1, Python 3.10 priority 2)
echo "🔧 Setting up Python alternatives..."
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 2>/dev/null || true
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2

# Auto-select Python 3.10 (highest priority)
echo "✅ Setting Python 3.10 as default..."
sudo update-alternatives --set python3 /usr/bin/python3.10

# Install pip for Python 3.10
echo "🔧 Installing pip for Python 3.10..."
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Verify installation
echo "🧪 Verifying installation..."
python3 --version
python3.10 --version
pip3 --version

echo "✅ Python 3.10 installation complete!"
echo "ℹ️  You may need to restart your Colab runtime for full compatibility"