#!/bin/bash

# Script to run the QwenDataset test with proper PYTHONPATH and logging
# Usage: ./run_test_qwen_dataset.sh
# Note: This script should be run from the video_SALMONN2_plus directory

set -e  # Exit on any error

# Get the current directory
CURRENT_DIR=$(pwd)
echo "============================================"
echo "Running QwenDataset Test"
echo "============================================"
echo "Current directory: $CURRENT_DIR"
echo "Setting PYTHONPATH to: ."
echo "Command: python qwenvl/data/test_qwen_dataset.py"
echo "Log file: qwenvl/data/test_qwen_dataset.log"
echo "============================================"

# Set PYTHONPATH and run the test, tee output to both stdout and log file
PYTHONPATH=. python qwenvl/data/test_qwen_dataset.py 2>&1 | tee qwenvl/data/test_qwen_dataset.log

echo "============================================"
echo "Test completed. Check qwenvl/data/test_qwen_dataset.log for full output."
echo "============================================"