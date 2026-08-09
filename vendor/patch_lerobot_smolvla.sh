#!/usr/bin/env bash
# Apply small compatibility patches to the external LeRobot SmolVLA checkout.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_SRC="${1:-${LEROBOT_ROOT:-${ROOT}/third_party/lerobot}}"

if [[ -L "${LEROBOT_SRC}" ]]; then
  LEROBOT_SRC="$(readlink -f "${LEROBOT_SRC}")"
fi

TARGET="${LEROBOT_SRC}/src/lerobot/processor/normalize_processor.py"

if [[ ! -f "${TARGET}" ]]; then
  echo "[patch_lerobot] missing normalize processor: ${TARGET}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[patch_lerobot] missing ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${TARGET}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

replacements = {
    "self._slice_stats_to_tensor(mean, std, tensor)": "self._slice_stats_to_tensor(mean, std, tensor=tensor)",
    "self._slice_stats_to_tensor(min_val, max_val, tensor)": "self._slice_stats_to_tensor(min_val, max_val, tensor=tensor)",
    "self._slice_stats_to_tensor(q01, q99, tensor)": "self._slice_stats_to_tensor(q01, q99, tensor=tensor)",
    "self._slice_stats_to_tensor(q10, q90, tensor)": "self._slice_stats_to_tensor(q10, q90, tensor=tensor)",
}

patched = 0
missing = []
for old, new in replacements.items():
    if old in text:
        text = text.replace(old, new)
        patched += 1
    elif new not in text:
        missing.append(old)

if missing:
    print("[patch_lerobot] unexpected normalize_processor.py; missing patterns:", file=sys.stderr)
    for pattern in missing:
        print("  " + pattern, file=sys.stderr)
    raise SystemExit(1)

if patched:
    path.write_text(text)
    print("[patch_lerobot] patched {} call(s) in {}".format(patched, path))
else:
    print("[patch_lerobot] already patched: {}".format(path))
PY

"${PYTHON_BIN}" -m py_compile "${TARGET}"
