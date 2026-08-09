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

exec "$@"
