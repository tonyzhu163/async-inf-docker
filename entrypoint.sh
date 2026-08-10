#!/usr/bin/env bash
# Container entrypoint: prepare the volume layout and write the LIBERO config
# that the pip `libero` package otherwise prompts for on first import.
#
# Container-side equivalent of
# scripts_by_author/tonyzhu163/server_prepare_storage.sh.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DATA_ROOT}/libero_config}"
LIBERO_DATASETS_PATH="${LIBERO_DATASETS_PATH:-${DATA_ROOT}/libero_datasets}"
export LIBERO_CONFIG_PATH LIBERO_DATASETS_PATH

mkdir -p \
  "${DATA_ROOT}"/{hf_cache,pip_cache,tmp,outputs,eval_logs,checkpoints,datasets} \
  "${ROBOSUITE_LOG_DIR:-${DATA_ROOT}/tmp/robosuite}" \
  "${LIBERO_CONFIG_PATH}" \
  "${LIBERO_DATASETS_PATH}"

# Rewritten every start: benchmark_root is an image path, the volume may carry a
# config from an older image.
python /usr/local/bin/write_libero_config.py

# Repo output dirs -> volume, matching the cluster's symlink layout, but only
# when a repo is actually mounted and the path is free.
if [[ -n "$(ls -A /workspace 2>/dev/null)" ]]; then
  for d in outputs eval_logs checkpoints; do
    if [[ -L "/workspace/${d}" || ! -e "/workspace/${d}" ]]; then
      ln -sfn "${DATA_ROOT}/${d}" "/workspace/${d}" 2>/dev/null || true
    fi
  done
fi

# --- job queue ----------------------------------------------------------------
# Runtime state, so it cannot live in an image layer. One group per PHYSICAL GPU
# at `parallel 1`: a task queued to gpuN can never share a device with another,
# which is the failure that silently stacked four eval arms on GPU 0.
#
# Count from `nvidia-smi -L`, not a hardcoded 4 -- the same image should work on
# an 8-GPU rental. Note this counts ENUMERABLE devices, so a card the driver
# cannot initialise is excluded automatically, which is the behaviour you want.
#
# Never fatal: this is convenience tooling, and the container must still start on
# a box with no GPU or no pueue.
if command -v pueued >/dev/null 2>&1; then
  pueued -d >/dev/null 2>&1 || true
  for _ in 1 2 3 4 5; do pueue status >/dev/null 2>&1 && break; sleep 0.4; done
  if pueue status >/dev/null 2>&1; then
    n_gpu="$(nvidia-smi -L 2>/dev/null | grep -c UUID || true)"
    for ((g = 0; g < ${n_gpu:-0}; g++)); do
      pueue group add "gpu${g}" >/dev/null 2>&1 || true
      pueue parallel 1 --group "gpu${g}" >/dev/null 2>&1 || true
    done
    echo "[entrypoint] pueue ready: ${n_gpu:-0} gpu group(s); queue with" \
         "'pueue add -g gpu0 -- <cmd>', chain with '--after <id>'"
  fi
fi

exec "$@"
