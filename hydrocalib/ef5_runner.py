"""Wrapper around the EF5 executable."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence


def run_ef5(control_file: str, ef5_executable: str = "./EF5/bin/ef5") -> subprocess.CompletedProcess:
    """Execute EF5 with the provided control file."""
    control_path = Path(control_file)
    if not control_path.exists():
        raise FileNotFoundError(f"Control file not found: {control_file}")
    try:
        result = subprocess.run(
            [ef5_executable, str(control_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"EF5 executable not found at {ef5_executable}") from exc
    return result
