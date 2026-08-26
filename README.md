# async-inf-smolvla — container image

Build environment for LIBERO async evaluation and SmolVLA training on RTX
4090/5090 (vast.ai). The image is the **environment only**; the research repo is mounted at
`/workspace` and data lives on the volume at `/data`.

This repo is public so Actions minutes and GHCR storage are free. It contains no
research code — only an env spec, three small support scripts, and the Dockerfile.

```bash
docker pull ghcr.io/tonyzhu163/async-inf-smolvla:blackwell
```

## Contents

| File | Role |
|---|---|
| `Dockerfile` | The image |
| `pip-overrides.txt` | Absolute version pins (uv overrides) |
| `entrypoint.sh` | Volume layout + LIBERO config on start |
| `write_libero_config.py` | Writes `config.yaml` without importing LIBERO |
| `vendor/` | Copied from the research repo — see "Keeping vendor/ in sync" |

## What is baked in

| Layer | Contents | Size |
|---|---|---|
| Base | `nvidia/cuda:12.8.1-base-ubuntu22.04` + EGL/OSMesa/GL, EGL vendor ICD | ~1.5 GB |
| Env | python 3.12, ffmpeg, torch 2.7.1+cu128, numpy 2.2.6 | ~11 GB |
| LeRobot | `BeneChen/lerobot@eval/smolvla`, patched, `-e .[smolvla,training,libero]` | ~0.5 GB |
| LIBERO | pip `libero` + the **full** asset tree overlaid from the official repo | ~1.5 GB |

~15 GB on disk, ~7 GB pushed, ~6 GB downloaded during the build — about 80% of
which is the CUDA stack PyTorch bundles, not anything project-specific.

Six things the cluster setup only gets right by hand, closed here at build time:

1. **`_slice_stats_to_tensor` patch** on the LeRobot fork.
2. **cu128 held on the first resolve.** Torch 2.7 is the first Blackwell-capable
   release and still supports the RTX 4090. uv `--overrides` keeps the whole
   install on one CUDA/PyTorch stack.
3. **`mujoco==3.3.2`**, by the same override rather than by install order. An
   eval-comparability pin, not a compatibility one: 3.8.1 renders darker floors and
   understates success rate. The build smoke asserts it.
4. **Full LIBERO assets.** The pip package ships 6 of 14 `stable_hope_objects`; a
   1-task smoke passes and then the full suite dies on `orange_juice.xml`.
5. **LIBERO config written before anything imports LIBERO.** `import libero.libero`
   prompts on stdin with no config on disk, hanging builds and batch jobs.
6. **robosuite's hardcoded `/tmp/robosuite.log`** redirected via a `usercustomize`
   shim, so a shared or full `/tmp` can't crash env construction.

## Run on vast

```bash
docker run --gpus all --shm-size=16g -v /data:/data -v $HOME/Revisiting-Async-Inf:/workspace -w /workspace -it ghcr.io/tonyzhu163/async-inf-smolvla:blackwell bash
```

`--shm-size=16g` is not optional: the default kills the LIBERO dataloader workers.

Verify GPUs and the renderer before launching anything real — neither is exercised by
the CI build, which has no GPU:

```bash
python /usr/local/bin/verify_gpu.py
```

The host driver must be 570.86 or newer. The verifier launches a real CUDA
kernel and creates a real EGL context; visibility checks alone do not prove a
5090-compatible install.

## Comparability boundary

The intentional change from the published environment is
`torch/torchvision/torchaudio 2.5.1/0.20.1 + cu121` to
`2.7.1/0.22.1 + cu128`. Trajectory-sensitive inputs remain fixed:

- LeRobot `8caf2c26322ae156d0aa733c65e8addeb626e138` plus the existing normalization patch
- `transformers==5.5.4`
- `hf-libero==0.1.4`, `robosuite==1.4.0`, `bddl==1.0.1`
- `mujoco==3.3.2`, `numpy==2.2.6`, `scipy==1.18.0`
- LIBERO asset tree `8f1084e3132a39270c3a13ebe37270a43ece2a01`

That asset commit is byte-identical to the 4090 image used for existing runs
(585 files, SHA-256 `68ebc7e4bf7b349e132c6c6cfe8bd484af6133a36ac8cbddffead2cd8d11cc66`).
A deterministic SmolVLA forward on a 4090 changed no action signs across 350
outputs when moving 2.5.1→2.7.1; mean absolute drift was 0.00193 and maximum
drift 0.0121. Compare pseudo-chunk and live-async arms inside this image; do not
mix old 2.5.1 controls with new 2.7.1 treatment rows.

Four 4090s have no NVLink and no P2P, so run four independent single-GPU shards pinned
with `CUDA_VISIBLE_DEVICES` rather than DDP.

## Fresh box bootstrap

Verified end-to-end on a new 4×4090 instance, 2026-08-19. Total wall time ~30 min,
dominated by the HF dataset pull. Run everything below inside the container over ssh.

```bash
# 1. Sanity: GPUs, mujoco pin, EGL. CI can't test this — no GPU there.
python /usr/local/bin/verify_gpu.py

# 2. HF downloads, immediately and in the background (HF_HOME already → /data/hf_cache).
mkdir -p /data/repos /data/envs /data/tmp
nohup hf download HuggingFaceVLA/libero --repo-type dataset > /data/tmp/dl_libero.log 2>&1 &
nohup sh -c 'hf download lerobot/pi05-libero && hf download lerobot/smolvla_base \
  && hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct' > /data/tmp/dl_models.log 2>&1 &

# 3. Research repo (from the workstation, not the box):
#    rsync -az --exclude=.git --exclude=outputs --exclude=pretrained_models \
#      Revisiting-Async-Inf/ <box>:/workspace/Revisiting-Async-Inf/

# 4. π0.5/vlash env — on the volume, not in the image (conflicting pins, see below).
git clone --branch sim/libero https://github.com/mit-han-lab/vlash /data/repos/vlash
UV=/opt/mamba/envs/lerobot-smolvla/bin/uv
export UV_CACHE_DIR=/data/pip_cache/uv CMAKE_POLICY_VERSION_MINIMUM=3.5
$UV venv --python 3.10 /data/envs/vlash --seed
$UV pip sync --python /data/envs/vlash/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match \
  /workspace/Revisiting-Async-Inf/requirements/vast/vlash-pi05.lock

# 5. Prove the rebuild: freeze must match the lock, CUDA and EGL must work.
$UV pip freeze --python /data/envs/vlash/bin/python | grep -vE '^(pip|wheel|setuptools)==|file:///' | sort \
  | diff - <(grep -E '^[a-zA-Z0-9_-]+==' /workspace/Revisiting-Async-Inf/requirements/vast/vlash-pi05.lock | sort)
/data/envs/vlash/bin/python -c "import torch; assert torch.cuda.is_available()"
MUJOCO_GL=egl /data/envs/vlash/bin/python -c "import mujoco; c=mujoco.GLContext(64,64); c.make_current(); print('egl ok')"
```

Expected landing point: torch 2.5.1+cu121, mujoco 3.3.7, numpy 1.24.4 in the vlash
env. A benign `GLContext.__del__` traceback at interpreter exit is normal. The
`lerobot-kinetix` env follows the same pattern from its own `.in`/lock.

The separate VLASH and Kinetix recipes above still carry their historical
CUDA locks and are not Blackwell-qualified by this image.

## Volume layout (300 GB)

Measured from the source cluster, 2026-08-09:

| Path | Contents | Measured |
|---|---|---|
| `/data/hf_cache` | `HuggingFaceVLA/libero` 33 GB, `SmolVLM2-500M-Video-Instruct` 6.7 GB, `smolvla_base` + `smolvla_libero` 1.8 GB | **42 GB** |
| `/data/hf_cache` | add `lerobot/pi05-libero` only if you run π0.5 | +8.8 GB |
| `/data/outputs` | training runs | **71 GB** |
| `/data/{eval_logs,checkpoints,datasets,tmp}` | logs, rollout video, scratch | 20–40 GB |

~135 GB steady-state, ~180 GB with π0.5. Start the HF download the moment the box is
up — it dwarfs the image pull.

The vast volume is machine-local: it pins the instance to that host and disappears with
it. Mirror promotion-gate checkpoints off-box.

## Lockfile

The image pins only what matters and lets uv resolve the rest. Every CI run uploads the
full resolution as an artifact (`lerobot-smolvla-docker.lock`); commit it to the
research repo under `requirements/` to make builds reproducible rather than merely
pinned.

## Keeping `vendor/` in sync

`vendor/` is copied from the private research repo and will drift if those files change:

| Here | Upstream |
|---|---|
| `vendor/lerobot-smolvla-cu128.yml` | Blackwell branch source of truth |
| `vendor/patch_lerobot_smolvla.sh` | `scripts_by_author/tonyzhu163/patch_lerobot_smolvla.sh` |
| `vendor/robosuite_logpatch/` | `scripts/robosuite_logpatch/` |

`pip-overrides.txt` must also stay consistent with the env yml's pins.

## Adding the other stacks

π0.5/vlash and vla-adapter want conflicting pins (mujoco 3.3.7 + numpy 1.24.4 for
vlash; tensorflow/dlimp for vla-adapter), so they belong in **separate tags**, not extra
conda envs here. Copy the Dockerfile, swap the env layer, keep the base and the `/data`
contract.
