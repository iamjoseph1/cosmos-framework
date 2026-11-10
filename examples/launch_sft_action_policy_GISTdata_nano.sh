#!/usr/bin/env bash
# Launch from the repository root with the training venv active.
set -euo pipefail
: "${GISTDATA_ACTION_BUNDLE:?Set the prepared bundle directory}"
: "${BASE_CHECKPOINT_PATH:?Set the matched base-model DCP directory}"
: "${WAN_VAE_PATH:?Set the matched Wan2.2 VAE path}"
if [[ ! -f "$BASE_CHECKPOINT_PATH/model/.metadata" && ! -f "$BASE_CHECKPOINT_PATH/.metadata" ]]; then
  echo "ERROR: BASE_CHECKPOINT_PATH must point to a converted base-model DCP checkpoint" >&2
  exit 1
fi
test -f "$WAN_VAE_PATH" || { echo "ERROR: WAN_VAE_PATH does not exist" >&2; exit 1; }
validation_args=(--bundle "$GISTDATA_ACTION_BUNDLE")
if [[ -n "${GISTDATA_TOKENIZER_FOLDER:-}" ]]; then
  validation_args+=(--tokenizer-folder "$GISTDATA_TOKENIZER_FOLDER" --tokenizer-model "${GISTDATA_TOKENIZER_MODEL:-nano}")
fi
python -m cosmos_framework.scripts.validate_GISTdata_action_data "${validation_args[@]}"
export IMAGINAIRE_OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs/train}"
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1
export WANDB_ENTITY="${WANDB_ENTITY:-inrol}"
# The recipe's job.wandb_mode controls logging; do not force disabled here.
sft_toml="${SFT_TOML_FILE:-examples/toml/sft_config/action_policy_GISTdata_nano.toml}"
test -f "$sft_toml" || { echo "ERROR: SFT_TOML_FILE does not exist" >&2; exit 1; }
torchrun --standalone --nproc-per-node="${NPROC_PER_NODE:-2}" \
  -m cosmos_framework.scripts.train \
  --sft-toml="$sft_toml" "$@"
