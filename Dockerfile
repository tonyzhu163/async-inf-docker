# syntax=docker/dockerfile:1.7
#
# lerobot-smolvla runnable env for LIBERO async eval + SmolVLA training.
#
# Mirrors scripts_by_author/tonyzhu163/setup_all.sh in the (private)
# Revisiting-Async-Inf repo, with the gaps that script leaves open closed:
#   1. LeRobot fork patched (_slice_stats_to_tensor keyword call)
#   2. cu121 held on the first resolve via uv overrides — no torch downgrade pass
#   3. mujoco pinned to the eval-comparable 3.3.2 by the same override
#   4. full LIBERO asset tree overlaid on the incomplete pip `libero` package
#   5. robosuite's hardcoded /tmp/robosuite.log redirected
#   6. LIBERO config written before anything imports LIBERO
#
# The research repo is NOT baked in — mount it at /workspace at run time.
# Data (HF cache, datasets, checkpoints, outputs) lives on the volume at /data.

FROM nvidia/cuda:12.1.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

# --- system: GL/EGL/OSMesa for MuJoCo headless, plus build + sync tools -------
# cmake and the mesa -dev headers are not optional: lerobot[libero] pulls
# hf-libero -> robomimic -> egl-probe, which ships only an sdist and builds a
# small EGL probe binary via CMake at install time.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake pkg-config ca-certificates curl git git-lfs rsync \
      bzip2 unzip less htop tmux vim-tiny \
      libegl1 libgles2 libglvnd0 libgl1 libglx0 libglfw3 libosmesa6 libosmesa6-dev \
      libegl1-mesa-dev libgl1-mesa-dev \
      libx11-6 libxext6 libxrender1 libxcursor1 libxinerama1 libxi6 libxrandr2 \
      patchelf

# EGL vendor ICD. The container toolkit injects libEGL_nvidia.so but not this
# JSON; without it MuJoCo's EGL init fails with the misleading
# "EGL driver does not support the PLATFORM_DEVICE extension".
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '{\n  "file_format_version" : "1.0.0",\n  "ICD" : {\n    "library_path" : "libEGL_nvidia.so.0"\n  }\n}\n' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# --- micromamba ---------------------------------------------------------------
# Kept (rather than going all-uv) because ffmpeg is a system library: Ubuntu
# 22.04 ships 4.4 and this stack expects the conda-forge 7.x build.
ENV MAMBA_ROOT_PREFIX=/opt/mamba \
    ENV_PREFIX=/opt/mamba/envs/lerobot-smolvla
RUN curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C /usr/local bin/micromamba

# --- conda env: python 3.12 + ffmpeg + cu121 torch pins -----------------------
# cu121 is deliberate: it is what every published number in the research repo
# was measured under, and the wheels carry sm_89 so they run on 4090s. Newer
# drivers are backward compatible.
COPY vendor/lerobot-smolvla-cu121.yml /tmp/env.yml
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    micromamba create -y -n lerobot-smolvla -f /tmp/env.yml && \
    micromamba clean -ay

ENV PATH=${ENV_PREFIX}/bin:${PATH}

# uv, for its override semantics (see the install layer) and because it resolves
# this dependency set in seconds rather than minutes.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install uv

# --- LeRobot fork -------------------------------------------------------------
# .git is kept: shallow, and removing it can break version resolution on an
# editable install.
ARG LEROBOT_REPO=https://github.com/BeneChen/lerobot.git
ARG LEROBOT_REF=eval/smolvla
RUN git clone --depth 1 --branch "${LEROBOT_REF}" "${LEROBOT_REPO}" /opt/lerobot

COPY vendor/patch_lerobot_smolvla.sh /tmp/patch_lerobot.sh
RUN PYTHON_BIN="${ENV_PREFIX}/bin/python" bash /tmp/patch_lerobot.sh /opt/lerobot

# Single-pass install. The fork's metadata declares torch>=2.7; the cluster's
# setup_all.sh copes by installing it and then downgrading via `conda env
# update`, which fetches ~2.5 GB of torch twice. uv overrides REPLACE a declared
# requirement (pip constraints only add to it, so `torch>=2.7` still wins and
# the resolve fails), so cu121 torch and mujoco 3.3.2 hold on the first pass and
# no wrong-version wheel is ever fetched.
#
# --index-strategy unsafe-best-match is required to see the +cu121 local
# versions on the PyTorch index alongside PyPI; uv's default first-index would
# never consider them.
COPY pip-overrides.txt /tmp/pip-overrides.txt
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python "${ENV_PREFIX}/bin/python" \
      --overrides /tmp/pip-overrides.txt \
      --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
      -e "/opt/lerobot[smolvla,training,libero]" \
      json_numpy rich

# --- complete the LIBERO asset tree -------------------------------------------
# The pip `libero` package ships 6 of 14 stable_hope_objects; any suite touching
# orange_juice/ketchup/cookies/... dies at env construction. Overlay the full
# tree from the official repo, then drop the clone in the same layer.
COPY write_libero_config.py /usr/local/bin/write_libero_config.py
RUN git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO /tmp/LIBERO_clone && \
    PKG_ROOT="$(python -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("libero").origin))')" && \
    rsync -a /tmp/LIBERO_clone/libero/libero/assets/ "${PKG_ROOT}/libero/assets/" && \
    rm -rf /tmp/LIBERO_clone && \
    test -f "${PKG_ROOT}/libero/assets/stable_hope_objects/orange_juice/orange_juice.xml" && \
    echo "stable_hope_objects: $(ls "${PKG_ROOT}/libero/assets/stable_hope_objects" | wc -l) objects"

# --- robosuite /tmp/robosuite.log shim ----------------------------------------
COPY vendor/robosuite_logpatch /opt/robosuite_logpatch

# --- runtime environment ------------------------------------------------------
ENV DATA_ROOT=/data \
    HF_HOME=/data/hf_cache \
    PIP_CACHE_DIR=/data/pip_cache \
    TMPDIR=/data/tmp \
    LIBERO_CONFIG_PATH=/data/libero_config \
    LIBERO_DATASETS_PATH=/data/libero_datasets \
    ROBOSUITE_LOG_DIR=/data/tmp/robosuite \
    PYTHONPATH=/opt/robosuite_logpatch \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    TOKENIZERS_PARALLELISM=false \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    NVIDIA_VISIBLE_DEVICES=all

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/write_libero_config.py

# Import smoke. Write a throwaway LIBERO config first — importing libero.libero
# with no config on disk prompts on stdin, which fails a non-interactive build.
RUN LIBERO_CONFIG_PATH=/tmp/libero_cfg LIBERO_DATASETS_PATH=/tmp/libero_data \
      python /usr/local/bin/write_libero_config.py && \
    LIBERO_CONFIG_PATH=/tmp/libero_cfg python - <<'PY'
import torch, torchvision, numpy, mujoco, lerobot, json_numpy, rich
from libero.libero import benchmark
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
print("numpy", numpy.__version__, "| mujoco", mujoco.__version__)
print("libero suites:", sorted(benchmark.get_benchmark_dict())[:4], "...")
assert torch.__version__.startswith("2.5.1"), torch.__version__
assert mujoco.__version__ == "3.3.2", mujoco.__version__
assert numpy.__version__ == "2.2.6", numpy.__version__
PY

# Freeze the resolution that actually shipped, for extraction by CI.
RUN uv pip freeze --python "${ENV_PREFIX}/bin/python" > /opt/lerobot-smolvla.lock && \
    wc -l /opt/lerobot-smolvla.lock

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
