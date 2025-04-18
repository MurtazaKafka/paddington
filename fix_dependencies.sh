#!/bin/bash

# Script to fix dependency conflicts in Paddington

echo "===== Fixing dependency conflicts for Paddington ====="

# Use the Python and pip from the virtual environment
VENV_DIR="./venv"
VENV_PIP="${VENV_DIR}/bin/pip"
VENV_PYTHON="${VENV_DIR}/bin/python"

# Check if the virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found at $VENV_DIR"
    echo "Please create and activate a virtual environment first"
    exit 1
fi

# Uninstall conflicting packages
echo "Removing conflicting packages..."
"$VENV_PIP" uninstall -y whisper numba

# Install correct versions
echo "Installing correct packages..."
"$VENV_PIP" install "openai-whisper==20231117"
"$VENV_PIP" install "numba==0.59.1"

echo "===== Dependencies fixed! ====="
echo "You can now run 'python run_paddington.py' to start the server." 