"""File helpers shared by the atomic writers (trial, probe and judge records; budget)."""
from __future__ import annotations

import os
import time
from pathlib import Path


def replace_with_retry(src: str | Path, dst: str | Path, attempts: int = 5,
                       delay: float = 0.2) -> None:
    """os.replace, retried on PermissionError.

    On Windows, OneDrive sync and antivirus briefly lock files (WinError 5 or 32), so
    an atomic replace can fail once or twice. Retries with a short sleep, then
    re-raises the last error.
    """
    for attempt in range(1, attempts + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(delay)
