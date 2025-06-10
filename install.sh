#!/bin/bash

echo "=============================================="
echo "  Contextual Music Crafter - Installation"
echo "=============================================="
echo

echo "[1/2] Checking for Python..."
# Prefer python3 if available
if command -v python3 &> /dev/null
then
    echo "Python 3 found."
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
# Fallback to python
elif command -v python &> /dev/null
then
    echo "Python found. We'll assume it's Python 3."
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    echo "ERROR: Python is not installed or not in your PATH." >&2
    echo "Please install Python 3.7+ to continue." >&2
    exit 1
fi
echo

echo "[2/2] Installing required Python packages..."
$PIP_CMD install -r requirements.txt
if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Package installation failed." >&2
    echo "Please ensure pip/pip3 is working correctly and try again." >&2
    exit 1
fi
echo

echo "=============================================="
echo "  Installation Successful!"
echo "=============================================="
echo
echo " IMPORTANT: Your next step is to open the"
echo " 'config.yaml' file and add your Google AI API key."
echo 