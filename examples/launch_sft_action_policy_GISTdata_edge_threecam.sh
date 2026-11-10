#!/usr/bin/env bash
# Fresh three-camera experiment with the same pinned Edge base checkpoint.
set -euo pipefail
export SFT_TOML_FILE=examples/toml/sft_config/action_policy_GISTdata_edge_threecam.toml
exec bash examples/launch_sft_action_policy_GISTdata_edge.sh "$@"
