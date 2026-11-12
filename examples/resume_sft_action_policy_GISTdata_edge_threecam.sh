#!/usr/bin/env bash
# Resume the existing 10000-step v2 job after storage has been freed.
set -euo pipefail
check_only=false
if [[ "${1:-}" == "--check-only" ]]; then
  check_only=true
  shift
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
source outputs/GISTdata_action/train_environment_threecam.sh

# GPU 0 was occupied when the failed job was inspected. GPU 1 has room
# alongside the head-camera job; override this UUID as the first argument.
export CUDA_VISIBLE_DEVICES="${1:-GPU-dfc2a4c3-e325-6cbe-2f0d-94ae57ca0743}"
export NPROC_PER_NODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python - <<'PY'
from pathlib import Path
import shutil
import subprocess

root = Path('outputs/train/rb_xhand_dish_pnp')
run = root / 'GISTdata_threecam_joint18/GISTdata_edge_threecam_v2'
checkpoint_root = run / 'checkpoints'
latest = (checkpoint_root / 'latest_checkpoint.txt').read_text().strip()
if not latest.startswith('iter_') or not latest[5:].isdigit():
    raise SystemExit(f'Unexpected latest checkpoint marker: {latest!r}')
checkpoint = checkpoint_root / latest
parts = ('model', 'optim', 'scheduler', 'trainer')
for part in parts:
    if not (checkpoint / part / '.metadata').is_file():
        raise SystemExit(f'Incomplete resume checkpoint: {checkpoint / part}')
checkpoint_bytes = sum(f.stat().st_size for f in checkpoint.rglob('*') if f.is_file())
step = int(latest[5:])
remaining_threecam = len(range((step // 2000 + 1) * 2000, 10001, 2000))
head_marker = root / 'GISTdata_head_joint18/GISTdata_edge_v2/checkpoints/latest_checkpoint.txt'
head_step = int(head_marker.read_text().strip().removeprefix('iter_')) if head_marker.exists() else 0
remaining_head = len(range((head_step // 1000 + 1) * 1000, 10001, 1000))
# Budget both jobs, including temporary checkpoint writes and a 20-GiB margin.
required = (remaining_threecam + remaining_head) * checkpoint_bytes + 20 * 1024**3
free = shutil.disk_usage(checkpoint_root).free
print(f'Resume checkpoint: {checkpoint} (optimizer step {step})', flush=True)
print(f'Disk: free={free / 1024**3:.1f} GiB, required={required / 1024**3:.1f} GiB', flush=True)
if free < required:
    raise SystemExit('Insufficient space for both jobs. Free storage before resuming; training was not launched.')

import os
memory = subprocess.check_output([
    'nvidia-smi', f'--id={os.environ["CUDA_VISIBLE_DEVICES"]}',
    '--query-gpu=memory.free', '--format=csv,noheader,nounits',
], text=True)
if int(memory.strip()) < 44 * 1024:
    raise SystemExit('Selected GPU has less than 44 GiB free. Select a GPU with more headroom.')
PY

if "$check_only"; then
  echo "Preflight passed; training was not launched."
  exit 0
fi

# Same job path lets the native DCP loader restore model, optimizer, scheduler
# and iteration together. Save every 2000 steps to reduce further disk use.
exec bash examples/launch_sft_action_policy_GISTdata_edge_threecam.sh -- \
  experiment=action_policy_GISTdata_edge_threecam_eval \
  model.config.parallelism.data_parallel_shard_degree=1 \
  trainer.max_iter=10000 \
  'scheduler.cycle_lengths=[10000]' \
  trainer.grad_accum_iter=8 \
  trainer.seed=42 \
  trainer.run_validation=true \
  trainer.run_validation_on_start=true \
  trainer.validation_iter=100 \
  checkpoint.save_iter=2000 \
  job.name=GISTdata_edge_threecam_v2
