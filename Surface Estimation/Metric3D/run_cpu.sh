#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source "$SCRIPT_DIR/../../.venv/bin/activate"

# Get the input path from arguments, default to "new test images"
# Using "$*" captures all arguments as a single string, helping with unquoted paths
INPUT_PATH="$*"
if [ -z "$INPUT_PATH" ]; then
    INPUT_PATH="new test images"
fi

# Run the CPU inference script
python run_model_cpu.py \
    'mono/configs/HourglassDecoder/vit.raft5.small.py' \
    --load-from ./weight/metric_depth_vit_small_800k.pth \
    --test_data_path "$INPUT_PATH" \
    --launcher None \
    --show-dir ./output

echo "Done. Results are in ./output"
