#!/usr/bin/env bash
# Run from the repository root after activating .venv-cosmos3-infer.
# Pass headcam|threecam, a current-input directory, a NEW output directory,
# then optional infer_GISTdata_actions flags (e.g. --actions-only).
set -euo pipefail
mode="${1:?Usage: bash examples/infer_GISTdata_edge.sh headcam|threecam INPUT_DIR NEW_OUTPUT [extra args]}"
input_dir="${2:?Set the current-input directory}"
output_dir="${3:?Set a new output directory}"
shift 3
case "$mode" in
  headcam) checkpoint_dir="${GISTDATA_CHECKPOINT_ROOT:-outputs/checkpoints/GISTdata}/headcam_edge_600" ;;
  threecam) checkpoint_dir="${GISTDATA_CHECKPOINT_ROOT:-outputs/checkpoints/GISTdata}/threecam_edge_600" ;;
  *) echo 'Expected headcam or threecam' >&2; exit 2 ;;
esac
export COSMOS_INTERNAL=0 COSMOS_TRAINING=0
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
args=(
  --image "$input_dir/head.png"
  --proprioception "$input_dir/proprioception.json"
  --instruction "$(cat "$input_dir/instruction.txt")"
  --checkpoint-path "$checkpoint_dir/checkpoints/iter_000000600"
  --config-file "$checkpoint_dir/config.local.yaml"
  --output "$output_dir" --seed 42 --num-steps 35 --guidance 3.0
)
if [[ "$mode" == threecam ]]; then
  args+=(--left-wrist-image "$input_dir/left_wrist.png" --right-wrist-image "$input_dir/right_wrist.png")
fi
# BF16 inference keeps the learned Edge generation-specific K norms intact.
exec python -m cosmos_framework.scripts.infer_GISTdata_actions "${args[@]}" "$@"
