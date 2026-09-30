import os, sys
from typing import Tuple
from datetime import datetime, timezone
from dataclasses import asdict

def progress_bar(pct, width=30):
    pct = max(0.0, min(100.0, float(pct)))
    filled = int(width * pct / 100.0)
    bar = "█" * filled + "·" * (width - filled)
    sys.stdout.write(f"\r[{bar}] {pct:6.2f}%")
    sys.stdout.flush()

def create_run_dir(
    base: str = None,
) -> str:
    """
    Create a unique, time-based run directory under `base`.

    Directory name encodes end point and enabled flags.
    """
    if base is None:
        from core.paths import DATA_DIR
        base = str(DATA_DIR / "recordings")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    name = f"{ts}"
    run_dir = os.path.join(base, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
