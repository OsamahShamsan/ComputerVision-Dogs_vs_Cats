#!/bin/bash
# ============================================
# Virtual Environment Activation Script
# ============================================
# This script ensures the virtual environment is activated
# Run this before any Python commands
# ============================================

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Path to virtual environment
VENV_PATH="$SCRIPT_DIR/venv"

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    echo "Please create it first: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Verify activation
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "✓ Virtual environment activated: $VIRTUAL_ENV"
    echo "✓ Python: $(which python)"
    echo "✓ Python version: $(python --version)"
else
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

