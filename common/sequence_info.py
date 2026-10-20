"""
info.json: one small provenance file per sequence (data/<Continent>/<City>/info.json).

{
  "author": "Jorge",                     # who downloaded and labelled it
  "source": "mapillary",                 # "mapillary" or "biel_glasses" (self-recorded in Barcelona)
  "sequence_id": "FHEALmdI9kgMRIWiDN3aHw",   # Mapillary sequence ("capture key"), null if unknown
  "source_url": "https://www.mapillary.com/app/?pKey=...",
  "note": "",
  "downloaded": "2026-10-08 19:54",
  "added_to_dataset": "2026-10-20",
  "in_paper": false                      # part of the dataset evaluated in the paper (arXiv:2605.11782)
}
"""
import json
from pathlib import Path

INFO_FILE = "info.json"
FIELDS = ["author", "source", "sequence_id", "source_url", "note", "downloaded", "added_to_dataset", "in_paper"]


def load_info(seq_dir):
    path = Path(seq_dir) / INFO_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_info(seq_dir, info):
    ordered = {k: info.get(k) for k in FIELDS}
    ordered.update({k: v for k, v in info.items() if k not in FIELDS})
    (Path(seq_dir) / INFO_FILE).write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_source_txt(seq_dir):
    """The hand-written source.txt some students add ("key: value" lines)."""
    path = Path(seq_dir) / "source.txt"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip().lower()] = value.strip()
    return {"sequence_id": out.get("sequence_id") or None, "source_url": out.get("source") or None,
            "note": out.get("note", ""), "downloaded": out.get("downloaded") or None}


def sequence_id_from_metadata(seq_dir):
    path = Path(seq_dir) / "metadata.json"
    if not path.exists():
        return None
    ids = {m.get("seq_id") for m in json.loads(path.read_text(encoding="utf-8"))} - {None}
    return ids.pop() if len(ids) == 1 else None


def image_ids(seq_dir):
    """Mapillary image IDs of a sequence (used to detect the same capture imported twice)."""
    path = Path(seq_dir) / "metadata.json"
    if not path.exists():
        return set()
    return {str(m["image_id"]) for m in json.loads(path.read_text(encoding="utf-8")) if m.get("image_id")}


def build_info(seq_dir, author=None, **overrides):
    """Existing info.json, completed with source.txt and metadata.json; `overrides` win."""
    info = {"author": None, "source": None, "sequence_id": None, "source_url": None, "note": "",
            "downloaded": None, "added_to_dataset": None, "in_paper": False}
    info.update({k: v for k, v in parse_source_txt(seq_dir).items() if v not in (None, "")})
    info.update({k: v for k, v in load_info(seq_dir).items() if v not in (None, "")})
    if not info["sequence_id"]:
        info["sequence_id"] = sequence_id_from_metadata(seq_dir)
    if not info["source"]:
        info["source"] = "mapillary" if (Path(seq_dir) / "metadata.json").exists() and image_ids(seq_dir) else "biel_glasses"
    if author:
        info["author"] = author
    info.update(overrides)
    return info
