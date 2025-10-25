import os
from typing import Tuple
from datetime import datetime, timezone
from dataclasses import asdict

def create_run_dir(
    base: str = "../data/recordings",
) -> str:
    """
    Create a unique, time-based run directory under `base`.

    Directory name encodes end point and enabled flags.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    name = f"{ts}"
    run_dir = os.path.join(base, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
