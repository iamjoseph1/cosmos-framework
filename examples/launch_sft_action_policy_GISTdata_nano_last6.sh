#!/usr/bin/env bash
# Source the server's train_environment.sh first to select physical GPU 1 and 3.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Select two healthy GPUs by UUID before launching}"
IFS=',' read -r -a selected_devices <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#selected_devices[@]}" -ne 2 || -z "${selected_devices[0]}" || -z "${selected_devices[1]}" || "${selected_devices[0]}" == "${selected_devices[1]}" ]]; then
  echo "ERROR: This recipe requires two distinct CUDA_VISIBLE_DEVICES entries" >&2
  exit 1
fi
export NPROC_PER_NODE=2
export SFT_TOML_FILE=examples/toml/sft_config/action_policy_GISTdata_nano_last6.toml
exec bash examples/launch_sft_action_policy_GISTdata_nano.sh "$@"
