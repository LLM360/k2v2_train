#!/usr/bin/env bash
# Merge SFT checkpoints to HF format, skipping ones already merged.
# Usage: bash merge_all_checkpoints_simple.sh

set -euo pipefail

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

# Gather step numbers from source and target
mapfile -t SOURCE_STEPS < <(ls -d "$BASE_DIR"/global_step_* 2>/dev/null | sed 's|.*/global_step_||' | sort -n)
mapfile -t DONE_STEPS   < <(ls -d "$TARGET_BASE_DIR"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n)

echo "Found ${#SOURCE_STEPS[@]} source checkpoints; ${#DONE_STEPS[@]} already present in target."

# Helper: does a merged target look complete?
looks_complete() {
  local d="$1"
  [[ -f "$d/config.json" ]] && { compgen -G "$d"/*.safetensors >/dev/null || compgen -G "$d"/pytorch_model*.bin >/dev/null; }
}

# Iterate through each checkpoint directory for merging
for step_num in "${SOURCE_STEPS[@]}"; do
  checkpoint_dir="$BASE_DIR/global_step_$step_num"
  target_dir="$TARGET_BASE_DIR/checkpoint-$step_num"

  echo "Checking global_step_$step_num -> $target_dir"

  if [[ -d "$target_dir" ]]; then
    if looks_complete "$target_dir"; then
      echo "  Skipped (already merged)"
      continue
    else
      echo "  Target exists but looks incomplete; re-merging..."
      rm -rf "$target_dir"
    fi
  fi

  echo "  Merging..."
  if python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$checkpoint_dir" \
        --target_dir "$target_dir"; then
    echo "  ✅ Success"
  else
    echo "  ❌ Failed"
  fi
done

echo "Merge completed! Results saved in: $TARGET_BASE_DIR"
