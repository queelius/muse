"""Runtime package installation helpers.

Keeps the CLI dependency graph slim: pull-a-model may install pip extras
on demand rather than forcing users to install everything upfront.
"""
from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)


def install_pip_extras(packages: list[str]) -> None:
    """Install pip packages that aren't already importable.

    Uses importlib.util.find_spec to check; skips silently if present.
    """
    missing = [p for p in packages if importlib.util.find_spec(_pkg_to_module(p)) is None]
    if not missing:
        return
    logger.info("installing pip packages: %s", missing)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *missing],
        check=True,
    )


def check_system_packages(packages: list[str]) -> list[str]:
    """Return the subset of system packages not found on PATH."""
    return [p for p in packages if shutil.which(p) is None]


def _pkg_to_module(pip_name: str) -> str:
    """Best-effort pip-name → importable-module mapping.

    Handles common mismatches (Pillow→PIL, beautifulsoup4→bs4).
    Falls back to the pip name itself.
    """
    mapping = {
        "Pillow": "PIL",
        "beautifulsoup4": "bs4",
        "opencv-python": "cv2",
        "scikit-learn": "sklearn",
        "huggingface_hub": "huggingface_hub",
    }
    # Strip extras like "misaki[en]" → "misaki"
    base = pip_name.split("[")[0].split(">=")[0].split("==")[0].strip()
    return mapping.get(base, base.replace("-", "_"))
