#!/bin/bash

# Script to update dependencies for Zonos TTS

echo "===== Updating dependencies for Zonos TTS ====="
echo "This script will update your Python packages to the versions required by Zonos."

# Ensure pip is up to date
pip install --upgrade pip

# Uninstall current packages that might conflict
echo "Removing potentially conflicting packages..."
pip uninstall -y transformers torch torchaudio numpy soundfile zonos

# Install specific versions of required packages
echo "Installing required package versions..."
pip install numpy>=2.2.2
pip install soundfile>=0.13.1
pip install torch>=2.5.1 torchaudio>=2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install transformers>=4.48.1
pip install accelerate sentencepiece protobuf

# Install Zonos directly from GitHub
echo "Installing Zonos from source..."
pip install git+https://github.com/Zyphra/Zonos.git

echo "===== All dependencies updated! ====="
echo "You can now run 'python run_paddington.py' to start the server." 