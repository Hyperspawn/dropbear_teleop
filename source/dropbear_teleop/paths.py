"""Where the robot model is. Everything this project needs is in this repository; ``$DROPBEAR_USD`` points the simulator
at another copy of the USD."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
USD_SHA256 = "45586414b065cd982d487cbd868fe982108b3b8ccec64d3dfcf629652ed8db0f"
"""SHA-256 of the robot USD this repository is calibrated against (``docs/CONTRACTS.md``)."""


def usd_path() -> Path:
    """The robot USD: ``$DROPBEAR_USD``, else ``assets/dropbear.usd``."""
    return Path(os.environ.get("DROPBEAR_USD") or REPO_ROOT / "assets" / "dropbear.usd").expanduser()
