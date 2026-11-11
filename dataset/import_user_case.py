"""
Imports PCPP-IRI's own ZED case-study recordings into data/user-case/<seq>/, in the same format
as the rest of the dataset: images_selected/ (evenly sampled keyframes) + metadata.json (GPS per
keyframe, from PCPP's own gt_manual.csv) + info.json. No ground_truth_labels.jsonl is written -
label the VQA questions afterwards the normal way (needs a desktop session, see README 5.4):

    python dataset/label_gt.py --continent user-case --city 01

The full ~100-1300 raw frames and the .svo2 recording stay in PCPP-IRI/data/zed/<seq>/ and are
not duplicated here - same policy as not re-downloading Mapillary's full-res images once a
sequence has its keyframes (they're not going anywhere, no need for a second copy).

    python dataset/import_user_case.py                   # all sequences found under PCPP_ZED_DIR
    python dataset/import_user_case.py --sequences 01 05  # just these
    python dataset/import_user_case.py --dry-run          # show what would happen, touch nothing
"""
import argparse
import csv
import json
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.paths import DATA_DIR, get_setting
from common.sequence_info import build_info, save_info

PCPP_ZED_DIR = Path.home() / "Documents" / "PCPP-IRI" / "data" / "zed"
NUM_KEYFRAMES = 20
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def load_zone_map():
    path = PCPP_ZED_DIR / "seq2zone.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_gt_positions(seq_dir):
    """frame_id (int) -> (lat, lon), from PCPP's own manually-refined localization."""
    path = seq_dir / "gt_manual.csv"
    positions = {}
    if not path.exists():
        return positions
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            positions[int(row["frame_id"])] = (float(row["gt_lat"]), float(row["gt_lon"]))
    return positions


def pick_keyframes(image_files, target_count=NUM_KEYFRAMES):
    """Evenly spaced sample, same stride approach as vqa/run.py and export_svo.py."""
    stride = max(1, len(image_files) // target_count)
    return image_files[::stride][:target_count]


def import_sequence(seq_name, author, dry_run=False):
    src_dir = PCPP_ZED_DIR / seq_name
    left_dir = src_dir / "images" / "left"
    if not left_dir.is_dir():
        print(f"SKIP {seq_name}: no images/left/ found")
        return False

    image_files = sorted(p for p in left_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_files:
        print(f"SKIP {seq_name}: no images in {left_dir}")
        return False

    keyframes = pick_keyframes(image_files)
    positions = load_gt_positions(src_dir)
    zone = load_zone_map().get(seq_name, "")
    missing_gps = sum(1 for img in keyframes if int(img.stem) not in positions)

    dest_dir = DATA_DIR / "user-case" / seq_name
    print(f"{seq_name} ({zone or 'unknown zone'}): {len(image_files)} frames -> "
          f"{len(keyframes)} keyframes -> {dest_dir}"
          + (f"  [{missing_gps} keyframe(s) missing GPS]" if missing_gps else ""))

    if dry_run:
        return True

    selected_dir = dest_dir / "images_selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for img in keyframes:
        shutil.copy2(img, selected_dir / img.name)
        lat, lon = positions.get(int(img.stem), (None, None))
        metadata.append({
            "filename": img.name,
            "latitude": lat,
            "longitude": lon,
            "captured_at": None,
            "compass_angle": None,
            "is_pano": False,
            "image_type": "regular",
            "position_source": "pcpp_zed_gt_manual",
        })
    (dest_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    info = build_info(
        dest_dir, author=author, source="biel_glasses", sequence_id=None, source_url=None,
        note=f"PCPP-IRI user-case study (zone: {zone})" if zone else "PCPP-IRI user-case study",
        downloaded=datetime.now().strftime("%Y-%m-%d %H:%M"),
        added_to_dataset=date.today().isoformat(), in_paper=False)
    save_info(dest_dir, info)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequences", nargs="+", help="Sequence folder names to import, e.g. 01 02 (default: all found)")
    parser.add_argument("--author", default=get_setting("VQA_AUTHOR"), help="Saved in info.json (default: VQA_AUTHOR in .env)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen; touch nothing")
    args = parser.parse_args()

    if not PCPP_ZED_DIR.is_dir():
        sys.exit(f"{PCPP_ZED_DIR} not found")

    sequences = args.sequences or sorted(
        p.name for p in PCPP_ZED_DIR.iterdir() if p.is_dir() and p.name.isdigit())
    if not sequences:
        sys.exit(f"No numbered sequence folders found under {PCPP_ZED_DIR}")

    done = 0
    for seq in sequences:
        if import_sequence(seq, args.author, args.dry_run):
            done += 1

    verb = "Would import" if args.dry_run else "Imported"
    print(f"\n{verb} {done}/{len(sequences)} sequence(s).")
    if not args.dry_run and done:
        print("Next, label the VQA questions for each (needs a desktop session):")
        print("  python dataset/label_gt.py --continent user-case --city <seq>")
