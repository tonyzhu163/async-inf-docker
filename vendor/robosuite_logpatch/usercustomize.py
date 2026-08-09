"""Auto-imported shim to make robosuite usable on shared clusters.

robosuite hard-codes ``logging.FileHandler("/tmp/robosuite.log")`` (see
robosuite/utils/log_utils.py) with no env override. On a shared node that path
is often already owned by another user, so opening it raises PermissionError and
crashes any process that imports robosuite (e.g. a LeRobot LIBERO env). It can
also fail with "No space left on device" on a node whose root fs is full.

Putting this directory on ``PYTHONPATH`` makes the interpreter import this module
during site initialisation (usercustomize). We wrap ``logging.FileHandler`` so
that the single hard-coded ``/tmp/robosuite.log`` path is redirected to a
writable directory (``$ROBOSUITE_LOG_DIR`` or the process temp dir); every other
FileHandler is untouched.
"""

import logging
import os
import tempfile

_ROBOSUITE_LOG = "/tmp/robosuite.log"
_OriginalFileHandler = logging.FileHandler


class _RobosuiteRedirectFileHandler(_OriginalFileHandler):
    def __init__(self, filename, *args, **kwargs):
        if str(filename) == _ROBOSUITE_LOG:
            target_dir = os.environ.get("ROBOSUITE_LOG_DIR") or tempfile.gettempdir()
            try:
                os.makedirs(target_dir, exist_ok=True)
            except OSError:
                target_dir = tempfile.gettempdir()
            filename = os.path.join(target_dir, "robosuite.log")
        super().__init__(filename, *args, **kwargs)


# Idempotent: only wrap once even if imported repeatedly.
if getattr(logging.FileHandler, "_robosuite_redirect", False) is False:
    _RobosuiteRedirectFileHandler._robosuite_redirect = True
    logging.FileHandler = _RobosuiteRedirectFileHandler
