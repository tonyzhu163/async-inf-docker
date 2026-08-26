"""Relax robosuite's EGL device guard for UUID-form CUDA_VISIBLE_DEVICES.

Upstream asserts `MUJOCO_EGL_DEVICE_ID in os.environ["CUDA_VISIBLE_DEVICES"]`
— a SUBSTRING test. When lanes are pinned by GPU UUID (necessary because EGL
enumeration order != PCI order on these boxes), the test passes only if the
EGL digit happens to occur in the UUID's hex string. On vast4 that silently
failed every cell on one lane (GPU1 uuid has no '2'; its EGL id is 2).

Idempotent. Prints what it did.
"""
import re
import sys

# Each env has its own robosuite copy; patching one does not patch another.
# Pass the target file (or an env root) as argv[1]; defaults to the mamba env.
DEFAULT = "/opt/mamba/envs/lerobot-smolvla/lib/python3.12/site-packages/robosuite/utils/binding_utils.py"
PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
if not PATH.endswith("binding_utils.py"):
    import glob as _glob
    hits = _glob.glob(PATH.rstrip("/") + "/lib/python*/site-packages/robosuite/utils/binding_utils.py")
    if not hits:
        print("no robosuite binding_utils.py under", PATH); sys.exit(2)
    PATH = hits[0]
print("target:", PATH)
MARK = "ROBOSUITE_TRUST_EGL_DEVICE_ID"  # newest guard; older UUID-only patches must upgrade

src = open(PATH).read()
if MARK in src:
    print("already patched")
    sys.exit(0)
OLD_GUARD = 'if not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"):'
if OLD_GUARD in src:
    src = src.replace(OLD_GUARD,
        'if (os.environ.get("ROBOSUITE_TRUST_EGL_DEVICE_ID") != "1"\n'
        '        and not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-")):', 1)
    open(PATH, "w").write(src)
    print("upgraded existing UUID-only guard to include ROBOSUITE_TRUST_EGL_DEVICE_ID")
    sys.exit(0)

# Match the whole assert statement, however it is wrapped across lines.
m = re.search(r"(?m)^(?P<ind>[ \t]*)assert\s+MUJOCO_EGL_DEVICE_ID\.isdigit\(\).*?\n(?=\S|\s*\n|\s*[a-zA-Z_])",
              src, re.S)
if not m:
    m = re.search(r"(?m)^(?P<ind>[ \t]*)assert\s+MUJOCO_EGL_DEVICE_ID\.isdigit\(\)[^\n]*\n(?:[ \t]+[^\n]*\n)*", src)
if not m:
    print("ASSERT NOT FOUND — no change")
    sys.exit(2)

stmt = m.group(0).rstrip("\n")
ind = m.group("ind")
guard = (
    f"{ind}# PATCHED 2026-08-24 (b): also honours ROBOSUITE_TRUST_EGL_DEVICE_ID=1, for\n"
    f"{ind}# numeric single-GPU lanes. CVD=0 with a probed EGL id of 2 fails the\n"
    f"{ind}# substring test even though 2 is the CORRECT egl index for cuda 0 on this\n"
    f"{ind}# box (cuda 0,1,2,3 -> egl 2,3,0,1). Set the opt-out only when the id came\n"
    f"{ind}# from an eglQueryDeviceAttribEXT(EGL_CUDA_DEVICE_NV) probe.\n"
    f"{ind}# PATCHED 2026-08-24: upstream tests MUJOCO_EGL_DEVICE_ID as a SUBSTRING of\n"
    f"{ind}# {MARK}, which is meaningless when we pin lanes by\n"
    f"{ind}# UUID (EGL order != PCI order here). It passed or failed by whether the EGL\n"
    f"{ind}# digit happened to appear in the UUID hex. The EGL device is still selected\n"
    f"{ind}# explicitly via MUJOCO_EGL_DEVICE_ID, so skip the check for UUID-form CVD.\n"
    f"{ind}if (os.environ.get(\"ROBOSUITE_TRUST_EGL_DEVICE_ID\") != \"1\"\n"
    f"{ind}        and not os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"\").startswith(\"GPU-\")):\n"
)
indented = "\n".join(("    " + ln) if ln.strip() else ln for ln in stmt.splitlines())
open(PATH, "w").write(src[: m.start()] + guard + indented + "\n" + src[m.end():])
print("patched")
