#!/usr/bin/env bash
# Merge SFT checkpoints to HF format, skipping ones already merged.
# Usage: bash merge_all_checkpoints_simple.sh

set -euo pipefail

########################################
# User options
########################################
# Optionally restrict which checkpoints to merge by step number.
# Examples:
#   SELECTED_STEPS=()                          # merge ALL available checkpoints (default behavior)
#   SELECTED_STEPS=(85000 90000 100000)        # merge ONLY these steps
#   SELECTED_STEPS=(global_step_85000 checkpoint-90000)  # prefixes ok
SELECTED_STEPS=(2400)

########################################
# Paths / env
########################################
BASE_DIR="/path/to/your/checkpoints"
TARGET_BASE_DIR="/path/to/your/merged-checkpoints"

# Create target directory
mkdir -p "$TARGET_BASE_DIR"

# Activate conda environment
source /path/to/your/conda.sh
conda activate your_env

# Switch to verl directory
cd /path/to/your/verl

########################################
# Discover source/target steps
########################################
mapfile -t SOURCE_STEPS < <(ls -d "$BASE_DIR"/global_step_* 2>/dev/null | sed 's|.*/global_step_||' | sort -n)
mapfile -t DONE_STEPS   < <(ls -d "$TARGET_BASE_DIR"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n)

echo "Found ${#SOURCE_STEPS[@]} source checkpoints; ${#DONE_STEPS[@]} already present in target."

if [[ ${#SOURCE_STEPS[@]} -eq 0 ]]; then
  echo "No source checkpoints found under: $BASE_DIR"
  exit 0
fi

########################################
# Normalize selection (if provided)
########################################
normalize_step() {
  local s="$1"
  s="${s#global_step_}"   # strip 'global_step_' if present
  s="${s#checkpoint-}"    # strip 'checkpoint-' if present
  echo "$s"
}

STEPS_TO_PROCESS=()
if [[ ${#SELECTED_STEPS[@]} -gt 0 ]]; then
  echo "Restricting to user-specified steps: ${SELECTED_STEPS[*]}"
  for raw in "${SELECTED_STEPS[@]}"; do
    step="$(normalize_step "$raw")"
    if [[ -d "$BASE_DIR/global_step_$step" ]]; then
      STEPS_TO_PROCESS+=("$step")
    else
      echo "  ⚠️  Requested step '$step' not found in $BASE_DIR; skipping."
    fi
  done
  if [[ ${#STEPS_TO_PROCESS[@]} -eq 0 ]]; then
    echo "No valid requested steps found. Nothing to do."
    exit 0
  fi
else
  # Default: all source steps
  STEPS_TO_PROCESS=("${SOURCE_STEPS[@]}")
fi

echo "Will process ${#STEPS_TO_PROCESS[@]} step(s): ${STEPS_TO_PROCESS[*]}"

########################################
# Helper: does a merged target look complete?
########################################
looks_complete() {
  local d="$1"
  [[ -f "$d/config.json" ]] && {
    compgen -G "$d"/*.safetensors >/dev/null || compgen -G "$d"/pytorch_model*.bin >/dev/null
  }
}

########################################
# Merge
########################################
for step_num in "${STEPS_TO_PROCESS[@]}"; do
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
