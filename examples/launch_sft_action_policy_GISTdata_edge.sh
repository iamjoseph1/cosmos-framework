#!/usr/bin/env bash
# Use the pinned Edge DCP and processor prepared by prepare_GISTdata_edge_checkpoint.
set -euo pipefail
: "${BASE_CHECKPOINT_PATH:?Set the prepared Cosmos3-Edge DCP directory}"
export GISTDATA_TOKENIZER_FOLDER
GISTDATA_TOKENIZER_FOLDER="$(python - <<'PY'
import json, os
from pathlib import Path
from cosmos_framework.utils.GISTdata_edge_contract import GISTDATA_EDGE_REPOSITORY, GISTDATA_EDGE_REVISION
receipt = Path(os.environ['BASE_CHECKPOINT_PATH']) / 'GISTdata_edge_base.json'
report = json.loads(receipt.read_text())
if (report['repository'], report['revision']) != (GISTDATA_EDGE_REPOSITORY, GISTDATA_EDGE_REVISION):
    raise ValueError('BASE_CHECKPOINT_PATH is not the matched GISTdata Edge release')
if not Path(report['snapshot']).is_dir():
    raise FileNotFoundError('Matched Edge processor snapshot is absent; rerun checkpoint preparation')
print(report['snapshot'])
PY
)"
export GISTDATA_TOKENIZER_MODEL=edge
export SFT_TOML_FILE=examples/toml/sft_config/action_policy_GISTdata_edge.toml
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
exec bash examples/launch_sft_action_policy_GISTdata_nano.sh "$@"
