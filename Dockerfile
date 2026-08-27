# syntax=docker/dockerfile:1.7
#
# lerobot-smolvla runnable env for LIBERO async eval + SmolVLA training.
#
# Mirrors scripts_by_author/tonyzhu163/setup_all.sh in the (private)
# Revisiting-Async-Inf repo, with the gaps that script leaves open closed:
#   1. LeRobot fork patched (_slice_stats_to_tensor keyword call)
#   2. one selected CUDA wheel profile held on the first resolve via uv overrides
#   3. mujoco pinned to the eval-comparable 3.3.2 by the same override
#   4. full LIBERO asset tree overlaid on the incomplete pip `libero` package
#   5. robosuite's hardcoded /tmp/robosuite.log redirected
#   6. robosuite's invalid UUID/EGL substring guard relaxed
#   7. LIBERO config written before anything imports LIBERO
#
# The research repo is NOT baked in — mount it at /workspace at run time.
# Data (HF cache, datasets, checkpoints, outputs) lives on the volume at /data.

ARG CUDA_BASE=12.8.1
FROM nvidia/cuda:${CUDA_BASE}-base-ubuntu22.04

ARG GPU_PROFILE=rtx5090
ARG TORCH_CUDA=cu128

COPY ENV_RELEASE /etc/async-inf-release
RUN printf '%s\n' "${GPU_PROFILE}/${TORCH_CUDA}" > /etc/async-inf-profile

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    ASYNC_INF_GPU_PROFILE=${GPU_PROFILE} \
    ASYNC_INF_TORCH_CUDA=${TORCH_CUDA}

# --- system: GL/EGL/OSMesa for MuJoCo headless, plus build + sync tools -------
# cmake and the mesa -dev headers are not optional: lerobot[libero] pulls
# hf-libero -> robomimic -> egl-probe, which ships only an sdist and builds a
# small EGL probe binary via CMake at install time.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake pkg-config ca-certificates curl git git-lfs rsync \
      openssh-server \
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

# --- conda env: python 3.12 + ffmpeg + selected torch profile -----------------
# The two builds differ only in the base image and CUDA wheel suffix. All
# benchmark/model pins come from the same template.
COPY vendor/lerobot-smolvla.yml /tmp/env.yml.in
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    sed "s/@TORCH_CUDA@/${TORCH_CUDA}/g" /tmp/env.yml.in > /tmp/env.yml && \
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
ARG LEROBOT_REF=8caf2c26322ae156d0aa733c65e8addeb626e138
RUN git init /opt/lerobot && cd /opt/lerobot && \
    git remote add origin "${LEROBOT_REPO}" && \
    git fetch --depth 1 origin "${LEROBOT_REF}" && \
    git checkout --detach FETCH_HEAD

COPY vendor/patch_lerobot_smolvla.sh /tmp/patch_lerobot.sh
RUN PYTHON_BIN="${ENV_PREFIX}/bin/python" bash /tmp/patch_lerobot.sh /opt/lerobot

# Single-pass install. The fork's metadata declares torch>=2.7; the cluster's
# setup_all.sh copes by installing it and then downgrading via `conda env
# update`, which fetches ~2.5 GB of torch twice. uv overrides REPLACE a declared
# requirement, so the CUDA and benchmark-sensitive pins hold on the first pass.
#
# --index-strategy unsafe-best-match is required to see the selected local
# versions on the PyTorch index alongside PyPI; uv's default first-index would
# never consider them.
#
# msgpack/websockets: the research repo's openpi websocket eval backend
# (openpi_client.websocket_client_policy -> msgpack_numpy -> msgpack) imports
# them at run time, and nothing in the lerobot extras declares either.
COPY pip-overrides.txt /tmp/pip-overrides.in
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    sed "s/@TORCH_CUDA@/${TORCH_CUDA}/g" /tmp/pip-overrides.in > /tmp/pip-overrides.txt && \
    uv pip install --python "${ENV_PREFIX}/bin/python" \
      --overrides /tmp/pip-overrides.txt \
      --index-strategy unsafe-best-match \
      --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
      -e "/opt/lerobot[smolvla,training,libero]" \
      json_numpy rich \
      msgpack==1.2.1 websockets==17.1

# robosuite 1.4 tests a host-global EGL index as a substring of
# CUDA_VISIBLE_DEVICES. That is meaningless for UUID-pinned lanes and caused
# one Vast4 lane to fail solely because its UUID did not contain digit "2".
COPY vendor/patch_robosuite_egl.py /tmp/patch_robosuite_egl.py
RUN python /tmp/patch_robosuite_egl.py "${ENV_PREFIX}" && \
    CUDA_VISIBLE_DEVICES=GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee \
    MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=disable \
    python -c 'import robosuite.utils.binding_utils'

# --- complete the LIBERO asset tree -------------------------------------------
# The pip `libero` package ships 6 of 14 stable_hope_objects; any suite touching
# orange_juice/ketchup/cookies/... dies at env construction. Overlay the full
# tree from the official repo, then drop the clone in the same layer.
COPY write_libero_config.py /usr/local/bin/write_libero_config.py
ARG LIBERO_REF=8f1084e3132a39270c3a13ebe37270a43ece2a01
RUN git init /tmp/LIBERO_clone && cd /tmp/LIBERO_clone && \
    git remote add origin https://github.com/Lifelong-Robot-Learning/LIBERO && \
    git fetch --depth 1 origin "${LIBERO_REF}" && \
    git checkout --detach FETCH_HEAD && \
    PKG_ROOT="$(python -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("libero").origin))')" && \
    rsync -a /tmp/LIBERO_clone/libero/libero/assets/ "${PKG_ROOT}/libero/assets/" && \
    rm -rf /tmp/LIBERO_clone && \
    test -f "${PKG_ROOT}/libero/assets/stable_hope_objects/orange_juice/orange_juice.xml" && \
    echo "stable_hope_objects: $(ls "${PKG_ROOT}/libero/assets/stable_hope_objects" | wc -l) objects"

# --- robosuite /tmp/robosuite.log shim ----------------------------------------
COPY vendor/robosuite_logpatch /opt/robosuite_logpatch

# --- job scheduling -----------------------------------------------------------
# A rented box has no scheduler, and hand-launching against N GPUs failed twice on
# 2026-08-10, both expensively. Four eval arms silently stacked on GPU 0 because
# each re-exported CUDA_VISIBLE_DEVICES=0 -- that variable is ABSOLUTE, so a child
# setting it overrides the parent's pin. A later batch was SIGKILLed eleven
# minutes in when the ssh session dropped; plain `nohup` does not survive that
# here. pueue removes both structurally: one group per GPU at `parallel 1` cannot
# double-book a device, and pueued is a real daemon.
#
# Pin the version. Asset naming changed between releases -- v4.0.4 publishes
# `pueue-x86_64-unknown-linux-musl`, older tags used `pueue-linux-x86_64` -- so an
# unpinned fetch breaks silently at some future tag.
ARG PUEUE_VERSION=v4.0.4
RUN set -eux; \
    base="https://github.com/Nukesor/pueue/releases/download/${PUEUE_VERSION}"; \
    for b in pueue pueued; do \
      curl -fsSL "${base}/${b}-x86_64-unknown-linux-musl" -o "/usr/local/bin/${b}"; \
      chmod +x "/usr/local/bin/${b}"; \
    done; \
    pueue --version; pueued --version

# nvitop into the SYSTEM python, not the conda env: it is an ops tool rather than
# a project dependency and must work whichever env is active. It attributes GPU
# memory PER PROCESS, which makes a double-booked device obvious at a glance
# instead of something inferred from aggregate nvidia-smi totals.
RUN pip3 install --no-cache-dir nvitop==1.7.1 && nvitop --version

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
COPY verify_gpu.py /usr/local/bin/verify_gpu.py
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/write_libero_config.py \
      /usr/local/bin/verify_gpu.py

# Import smoke. Write a throwaway LIBERO config first — importing libero.libero
# with no config on disk prompts on stdin, which fails a non-interactive build.
RUN LIBERO_CONFIG_PATH=/tmp/libero_cfg LIBERO_DATASETS_PATH=/tmp/libero_data \
      python /usr/local/bin/write_libero_config.py && \
    LIBERO_CONFIG_PATH=/tmp/libero_cfg python - <<'PY'
import importlib.metadata as md
import os
import torch, torchvision, numpy, mujoco, lerobot, json_numpy, rich
import msgpack, websockets
from libero.libero import benchmark
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
print("numpy", numpy.__version__, "| mujoco", mujoco.__version__)
print("libero suites:", sorted(benchmark.get_benchmark_dict())[:4], "...")
suffix = os.environ["ASYNC_INF_TORCH_CUDA"]
cuda = f"{suffix[2:4]}.{suffix[4:]}"
assert torch.__version__ == f"2.7.1+{suffix}", torch.__version__
assert torchvision.__version__ == f"0.22.1+{suffix}", torchvision.__version__
assert torch.version.cuda == cuda, torch.version.cuda
assert mujoco.__version__ == "3.3.2", mujoco.__version__
assert numpy.__version__ == "2.2.6", numpy.__version__
for package, version in {
    "robosuite": "1.4.0",
    "bddl": "1.0.1",
    "hf-libero": "0.1.4",
    "scipy": "1.18.0",
    "transformers": "5.5.4",
    "msgpack": "1.2.1",
    "websockets": "17.1",
}.items():
    assert md.version(package) == version, (package, md.version(package))
PY

# Freeze the resolution that actually shipped, for extraction by CI.
RUN uv pip freeze --python "${ENV_PREFIX}/bin/python" > /opt/lerobot-smolvla.lock && \
    wc -l /opt/lerobot-smolvla.lock

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
