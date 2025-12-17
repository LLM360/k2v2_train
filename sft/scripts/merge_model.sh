#!/bin/bash

# Simple script to merge all checkpoints automatically
# Usage: bash merge_all_checkpoints_simple.sh

# Set base paths
BASE_DIR="/path/to/your/checkpoints"
TARGET_BASE_DIR="/path/to/your/merged-checkpoints"

# Create target directory
mkdir -p "$TARGET_BASE_DIR"

# Activate conda environment
source /path/to/your/conda.sh
conda activate your_env

# Switch to verl directory
cd /path/to/your/verl

# Get all checkpoint directories and sort them
CHECKPOINT_DIRS=($(ls -d "$BASE_DIR"/global_step_* 2>/dev/null | sort -V))

echo "Found ${#CHECKPOINT_DIRS[@]} checkpoint directories"

# Iterate through each checkpoint directory for merging
for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"; do
    step_num=$(basename "$checkpoint_dir" | sed 's/global_step_//')
    target_dir="$TARGET_BASE_DIR/checkpoint-$step_num"

    echo "Merging global_step_$step_num -> checkpoint-$step_num"

    # Skip if target directory already exists
    if [ -d "$target_dir" ]; then
        echo "  Skipped (already exists)"
        continue
    fi

    # Execute merge command
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$checkpoint_dir" \
        --target_dir "$target_dir"

    if [ $? -eq 0 ]; then
        echo "  :white_check_mark: Success"
    else
        echo "  :x: Failed"
    fi
done

echo "Merge completed! Results saved in: $TARGET_BASE_DIR"