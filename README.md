# async-inf-smolvla — container image

Build environment for LIBERO async evaluation and SmolVLA training on RTX
4090/5090 (vast.ai). The image is the **environment only**; the research repo is mounted at
`/workspace` and data lives on the volume at `/data`.

This repo is public so Actions minutes and GHCR storage are free. It contains no
research code — only an env spec, three small support scripts, and the Dockerfile.

This image is the `lerobot-smolvla` **LIBERO benchmark + Torch-policy base** of
the research repo's `forward-v1` release. It has two accelerator profiles with
identical model/benchmark pins:

| tag | target | PyTorch wheel | system CUDA base |
|---|---|---|---|
| `rtx4090` / `ada` | RTX 4090 | 2.7.1+cu126 | 12.1.1 (keeps Vast2's 550 driver usable) |
| `rtx5090` / `blackwell` | RTX 5090 | 2.7.1+cu128 | 12.8.1 |

The benchmark/backend compatibility map lives in
[`requirements/ENV_MAP.md`](https://github.com/tonyzhu163/Revisiting-Async-Inf/blob/main/requirements/ENV_MAP.md);
the extension locks remain in the research repo.

```bash
docker pull ghcr.io/tonyzhu163/async-inf-smolvla:rtx4090
docker pull ghcr.io/tonyzhu163/async-inf-smolvla:rtx5090
```

Use the immutable digest recorded in the research repo for evidence runs; the
human-readable tags are deployment aliases. There is deliberately no `latest`
tag: choosing Ada or Blackwell is part of the run identity.

## Contents

| File | Role |
|---|---|
| `ENV_RELEASE` | Logical release identity baked into `/etc/async-inf-release` |
| `Dockerfile` | The image |
| `pip-overrides.txt` | Absolute version pins (uv overrides) |
| `entrypoint.sh` | Volume layout + LIBERO config on start |
| `write_libero_config.py` | Writes `config.yaml` without importing LIBERO |
| `vendor/` | Copied from the research repo — see "Keeping vendor/ in sync" |

## What is baked in

| Layer | Contents | Size |
|---|---|---|
| Base | profile-selected NVIDIA CUDA base + EGL/OSMesa/GL, EGL vendor ICD | ~1.5 GB |
| Env | python 3.12, ffmpeg, torch 2.7.1+cu126/cu128, numpy 2.2.6 | ~11 GB |
| LeRobot | `BeneChen/lerobot@eval/smolvla`, patched, `-e .[smolvla,training,libero]` | ~0.5 GB |
| LIBERO | pip `libero` + the **full** asset tree overlaid from the official repo | ~1.5 GB |

~15 GB on disk, ~7 GB pushed, ~6 GB downloaded during the build — about 80% of
which is the CUDA stack PyTorch bundles, not anything project-specific.

Seven things the cluster setup only gets right by hand, closed here at build time:

1. **`_slice_stats_to_tensor` patch** on the LeRobot fork.
2. **One CUDA wheel profile held on the first resolve.** uv `--overrides`
   prevents a mixed cu126/cu128 dependency graph.
3. **`mujoco==3.3.2`**, by the same override rather than by install order. An
   eval-comparability pin, not a compatibility one: 3.8.1 renders darker floors and
   understates success rate. The build smoke asserts it.
4. **Full LIBERO assets.** The pip package ships 6 of 14 `stable_hope_objects`; a
   1-task smoke passes and then the full suite dies on `orange_juice.xml`.
5. **LIBERO config written before anything imports LIBERO.** `import libero.libero`
   prompts on stdin with no config on disk, hanging builds and batch jobs.
6. **robosuite's hardcoded `/tmp/robosuite.log`** redirected via a `usercustomize`
   shim, so a shared or full `/tmp` can't crash env construction.
7. **robosuite's UUID/EGL substring guard** relaxed. UUID-pinned lanes still
   select an explicit probed EGL device instead of passing or failing according
   to whether that device digit happens to appear in the UUID text.

## Run on vast

```bash
IMAGE=ghcr.io/tonyzhu163/async-inf-smolvla:rtx4090  # use rtx5090 on Blackwell
docker run --gpus all --shm-size=16g -v /data:/data -v $HOME/Revisiting-Async-Inf:/workspace -w /workspace -it "$IMAGE" bash
```

`--shm-size=16g` is not optional: the default kills the LIBERO dataloader workers.

Verify GPUs and the renderer before launching anything real — neither is exercised by
the CI build, which has no GPU:

```bash
python /usr/local/bin/verify_gpu.py
```

The verifier launches a real CUDA kernel and creates a real EGL context;
visibility checks alone do not prove either profile works on a host driver.

## Comparability boundary

The intentional change from the published environment is
`torch/torchvision/torchaudio 2.5.1/0.20.1 + cu121` to
`2.7.1/0.22.1 + cu126` (4090) or `+cu128` (5090). Trajectory-sensitive inputs remain fixed:

- LeRobot `8caf2c26322ae156d0aa733c65e8addeb626e138` plus the existing normalization patch
- `transformers==5.5.4`
- `hf-libero==0.1.4`, `robosuite==1.4.0`, `bddl==1.0.1`
- `mujoco==3.3.2`, `numpy==2.2.6`, `scipy==1.18.0`
- LIBERO asset tree `8f1084e3132a39270c3a13ebe37270a43ece2a01`

That asset commit is byte-identical to the 4090 image used for existing runs
(585 files, SHA-256 `68ebc7e4bf7b349e132c6c6cfe8bd484af6133a36ac8cbddffead2cd8d11cc66`).
A deterministic SmolVLA forward on a 4090 changed no action signs across 350
outputs when moving 2.5.1→2.7.1; mean absolute drift was 0.00193 and maximum
drift 0.0121. Compare pseudo-chunk and live-async arms inside one accelerator
profile; do not mix old 2.5.1 controls with new 2.7.1 treatment rows.

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

# 4. Build only the role extensions you need. Canonical profile commands,
# source revisions, and bounded smokes live in requirements/vast/README.md and
# requirements/domino-pi05-uv.md in the research repo.
```

The frozen VLASH reproduction appliance uses the selected Torch profile while
checkpoint/action parity is completed for absorption into this base. Kinetix
inherits this image's Torch and uses JAX 0.5.3 on CPU by default, leaving the
selected GPU to Torch; standalone GPU-JAX Kinetix is a separate service/process.
A benign `GLContext.__del__` traceback at interpreter exit is normal.

On an existing Vast template whose persistent volume is mounted at
`/workspace`, keep the instance: migrate the small current `/data` contents
once, then point `/data` at `/workspace`. New templates should mount the volume
at `/data` directly. Do not maintain two caches.

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
full resolutions as `lerobot-smolvla-cu126.lock` and
`lerobot-smolvla-cu128.lock` artifacts. Qualified copies are committed in the
research repo under `requirements/`.

## Keeping `vendor/` in sync

`vendor/` is copied from the private research repo and will drift if those files change:

| Here | Upstream |
|---|---|
| `vendor/lerobot-smolvla.yml` + `pip-overrides.txt` | research repo's cu126/cu128 profiles (same pins except CUDA triplet) |
| `vendor/patch_lerobot_smolvla.sh` | `scripts_by_author/tonyzhu163/patch_lerobot_smolvla.sh` |
| `vendor/robosuite_logpatch/` | `scripts/robosuite_logpatch/` |
| `vendor/patch_robosuite_egl.py` | `scripts/maintenance/patch_robosuite_egl.py` |

`pip-overrides.txt` must also stay consistent with the env yml's pins.

## Other runtime roles

Kinetix is an overlay on this base. OpenPI and DOMINO are separate cooperating
services, while VLASH is an isolated reproduction environment. Their qualified
locks stay separate; do not add their conflicting packages to this base image.
