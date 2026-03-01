#!/usr/bin/env python3
"""Compatibility wrapper for the camera trap CLI."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "apps" / "camera_trap" / "cli" / "dap3_cli.py"
    runpy.run_path(str(target), run_name="__main__")
