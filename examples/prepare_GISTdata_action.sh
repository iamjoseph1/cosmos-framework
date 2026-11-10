#!/usr/bin/env bash
# Usage: bash examples/prepare_GISTdata_action.sh /path/to/GISTdata /path/to/new_bundle
# Requires a Cosmos training environment, or Python+numpy for preparation alone.
# Input: 100 bowl + 100 mug trajectories and completed three-camera recordings.
# Output: portable head MP4s, measured states, absolute commands, instructions,
# grouped train/val/all window manifests, and structural/sample validation.
set -euo pipefail
source_root="${1:?Pass the GISTdata source directory}"
bundle_root="${2:?Pass a NEW prepared output directory}"
python -m cosmos_framework.scripts.prepare_GISTdata_action_data \
  --source "$source_root" --output "$bundle_root" --expected-episodes 200
python -m cosmos_framework.scripts.validate_GISTdata_action_data --bundle "$bundle_root"
