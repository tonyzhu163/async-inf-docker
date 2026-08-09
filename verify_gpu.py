#!/usr/bin/env python
"""First-boot check for the two things CI cannot verify (its runners have no GPU).

Deliberately launches a real kernel and creates a real EGL context. Checking
`torch.cuda.is_available()` or `device_count()` is NOT enough: on an
architecture the wheels lack kernels for (e.g. Blackwell sm_120 under cu121)
both report success and only the kernel launch fails.

    docker run --gpus all --rm <image> python /usr/local/bin/verify_gpu.py
"""

import sys

import torch

print(f"torch {torch.__version__} | devices {torch.cuda.device_count()}")
if not torch.cuda.is_available():
    sys.exit("FAIL: no CUDA device visible (missing --gpus all?)")

for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    cap = ".".join(map(str, torch.cuda.get_device_capability(i)))
    # The real test: this raises "no kernel image is available" on an
    # unsupported arch, where the availability checks above pass fine.
    x = torch.randn(256, 256, device=f"cuda:{i}")
    ok = bool((x @ x).isfinite().all())
    print(f"  cuda:{i} {name} sm_{cap.replace('.', '')} matmul={'ok' if ok else 'FAILED'}")
    if not ok:
        sys.exit(f"FAIL: non-finite matmul on cuda:{i}")

import mujoco

print(f"mujoco {mujoco.__version__}")
if mujoco.__version__ != "3.3.2":
    sys.exit(f"FAIL: mujoco {mujoco.__version__}, expected 3.3.2 (eval comparability pin)")

# Actually initialise EGL rather than just importing it.
from mujoco.egl import GLContext

ctx = GLContext(64, 64)
ctx.make_current()
print("egl: context created and made current")

print("OK: kernels launch, mujoco pinned, EGL renders")
