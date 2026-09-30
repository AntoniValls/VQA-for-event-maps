"""
Central paths and settings, so every script works regardless of the current directory.

The dataset folder defaults to <repo>/data and can be moved elsewhere by setting
VQA_DATA_DIR (environment variable or .env file at the repository root).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
PROMPTS_PATH = ROOT / "inout" / "vqa_prompts.json"

CONTINENTS = ["Africa", "America", "Asia", "Europe", "Oceania"]
MODELS = ["qwen-vl", "llava", "instructblip", "vilt"]


def get_setting(name, default=None):
    """Read a setting from the environment, falling back to the (git-ignored) .env file."""
    value = os.environ.get(name)
    if value:
        return value
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            key, _, val = line.partition("=")
            if key.strip() == name:
                return val.strip().strip('"').strip("'")
    return default


DATA_DIR = Path(get_setting("VQA_DATA_DIR") or ROOT / "data").expanduser().resolve()


def sequence_dir(continent, city):
    """data/<Continent>/<City>"""
    return DATA_DIR / continent / city


def list_sequences(continent=None, city=None, require_gt=True):
    """
    Discover sequences in the data folder as (continent, city) tuples.
    Only folders with a ground_truth_labels.jsonl are returned when require_gt=True.
    """
    sequences = []
    for cont in CONTINENTS:
        if continent and cont != continent:
            continue
        cont_dir = DATA_DIR / cont
        if not cont_dir.is_dir():
            continue
        for seq in sorted(p for p in cont_dir.iterdir() if p.is_dir()):
            if city and seq.name != city:
                continue
            if require_gt and not (seq / "ground_truth_labels.jsonl").exists():
                continue
            sequences.append((cont, seq.name))
    return sequences


def record_path(path):
    """Path string stored in the jsonl files (kept as '../data/<Continent>/<City>/...' for all records)."""
    path = Path(path).resolve()
    try:
        return str(Path("../data") / path.relative_to(DATA_DIR))
    except ValueError:
        return str(path)
