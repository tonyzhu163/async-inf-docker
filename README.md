# async-inf-smolvla — container image

Build environment for LIBERO async evaluation and SmolVLA training on 4×4090
(vast.ai). The image is the **environment only**; the research repo is mounted at
`/workspace` and data lives on the volume at `/data`.

This repo is public so Actions minutes and GHCR storage are free. It contains no
research code — only an env spec, three small support scripts, and the Dockerfile.

```bash
docker pull ghcr.io/tonyzhu163/async-inf-smolvla:latest
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
| Base | `nvidia/cuda:12.1.1-base-ubuntu22.04` + EGL/OSMesa/GL, EGL vendor ICD | ~1.5 GB |
| Env | python 3.12, ffmpeg, torch 2.5.1+cu121, numpy 2.2.6 | ~11 GB |
| LeRobot | `BeneChen/lerobot@eval/smolvla`, patched, `-e .[smolvla,training,libero]` | ~0.5 GB |
| LIBERO | pip `libero` + the **full** asset tree overlaid from the official repo | ~1.5 GB |

~15 GB on disk, ~7 GB pushed, ~6 GB downloaded during the build — about 80% of
which is the CUDA stack PyTorch bundles, not anything project-specific.

Six things the cluster setup only gets right by hand, closed here at build time:

1. **`_slice_stats_to_tensor` patch** on the LeRobot fork.
2. **cu121 held on the first resolve.** The fork's metadata declares `torch>=2.7`.
   `setup_all.sh` copes by installing it and then downgrading through `conda env
   update` — fetching ~2.5 GB of torch twice. pip constraints can't fix this (they're
   additive, so `torch>=2.7` still wins and the resolve fails), but uv `--overrides`
   *replace* a declared requirement, so the wrong wheel is never fetched.
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
docker run --gpus all --shm-size=16g -v /data:/data -v $HOME/Revisiting-Async-Inf:/workspace -w /workspace -it ghcr.io/tonyzhu163/async-inf-smolvla:latest bash
```

`--shm-size=16g` is not optional: the default kills the LIBERO dataloader workers.

Verify GPUs and the renderer before launching anything real — neither is exercised by
the CI build, which has no GPU:

```bash
python -c "import torch, mujoco.egl; print(torch.__version__, torch.cuda.device_count(), 'egl ok')"
```

Four 4090s have no NVLink and no P2P, so run four independent single-GPU shards pinned
with `CUDA_VISIBLE_DEVICES` rather than DDP.

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
| `vendor/lerobot-smolvla-cu121.yml` | `environment/lerobot-smolvla-cu121.yml` |
| `vendor/patch_lerobot_smolvla.sh` | `scripts_by_author/tonyzhu163/patch_lerobot_smolvla.sh` |
| `vendor/robosuite_logpatch/` | `scripts/robosuite_logpatch/` |

`pip-overrides.txt` must also stay consistent with the env yml's pins.

## Adding the other stacks

π0.5/vlash and vla-adapter want conflicting pins (mujoco 3.3.7 + numpy 1.24.4 for
vlash; tensorflow/dlimp for vla-adapter), so they belong in **separate tags**, not extra
conda envs here. Copy the Dockerfile, swap the env layer, keep the base and the `/data`
contract.
