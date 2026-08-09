#!/usr/bin/env python
"""Write $LIBERO_CONFIG_PATH/config.yaml without importing the libero package.

`import libero.libero` prompts on stdin when no config exists, which hangs a
docker build and any batch job. Resolve the package directory from the module
spec instead, so the config is on disk before anything imports LIBERO.

Container-side equivalent of the write_libero_config() step in
scripts_by_author/tonyzhu163/server_prepare_storage.sh.
"""

import importlib.util
import os
import sys
from pathlib import Path

spec = importlib.util.find_spec("libero")
if spec is None or not spec.origin:
    sys.exit("[libero-config] libero package not found")

bench_root = Path(spec.origin).parent / "libero"
if not (bench_root / "bddl_files").is_dir():
    sys.exit(f"[libero-config] no bddl_files under {bench_root}")

config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", "/data/libero_config"))
datasets = os.environ.get("LIBERO_DATASETS_PATH", "/data/libero_datasets")
config_dir.mkdir(parents=True, exist_ok=True)

(config_dir / "config.yaml").write_text(
    f"assets: {bench_root}/assets\n"
    f"bddl_files: {bench_root}/bddl_files\n"
    f"benchmark_root: {bench_root}\n"
    f"datasets: {datasets}\n"
    f"init_states: {bench_root}/init_files\n"
)
print(f"[libero-config] wrote {config_dir / 'config.yaml'} (benchmark_root={bench_root})")
