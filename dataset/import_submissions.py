"""
Moves students' sequences from students_data/ into the dataset (data/<Continent>/<City>/),
recording who made each one in its info.json.

    python dataset/import_submissions.py students_data --dry-run     # show what would happen
    python dataset/import_submissions.py students_data               # import every student
    python dataset/import_submissions.py students_data/Jorge         # import one student

Expected layout (what extracting a student's <Continent>_<City>.tar.gz gives):
    students_data/<Author>/<Continent>_<City>/data/<Continent>/<City>/ground_truth_labels.jsonl ...
The author is the folder right below students_data/ (or --author).

For each sequence:
  - skipped if the same Mapillary images are already in the dataset (same capture imported twice)
  - renamed to the next free number if data/<Continent>/<City> is taken (Singapore -> Singapore2)
  - info.json written (author, Mapillary sequence id and link, dates); a student's source.txt is folded into it
  - moved into data/, and the emptied student folders are removed

Tag sequences that are already in data/ but have no info.json yet (e.g. your own):
    python dataset/import_submissions.py --tag-existing Antoni [--in-paper]
"""
import argparse
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.paths import CONTINENTS, DATA_DIR
from common.sequence_info import INFO_FILE, build_info, image_ids, load_info, save_info


def find_submissions(root):
    """Sequence folders (holding ground_truth_labels.jsonl) under root."""
    return sorted(p.parent for p in Path(root).rglob("ground_truth_labels.jsonl") if p.parent.name != "results")


def infer_author(seq_dir):
    parts = Path(seq_dir).resolve().parts
    if "students_data" in parts:
        i = parts.index("students_data")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def dataset_sequences():
    """Every sequence folder already in data/<Continent>/ (one with images, labels or metadata)."""
    out = []
    for cont in CONTINENTS:
        if (DATA_DIR / cont).is_dir():
            out += [p for p in sorted((DATA_DIR / cont).iterdir()) if p.is_dir() and any(
                (p / f).exists() for f in ("images", "ground_truth_labels.jsonl", "metadata.json"))]
    return out


def free_name(continent, city, taken):
    """City, or the next free numbered name: Singapore -> Singapore2, Lisbon1 -> Lisbon2."""
    if not (DATA_DIR / continent / city).exists() and (continent, city) not in taken:
        return city
    base = re.sub(r"\d+$", "", city)
    n = 2
    while (DATA_DIR / continent / f"{base}{n}").exists() or (continent, f"{base}{n}") in taken:
        n += 1
    return f"{base}{n}"


def review_status(seq_dir):
    path = Path(seq_dir) / "review.json"
    if not path.exists():
        return "not reviewed"
    status = json.loads(path.read_text(encoding="utf-8")).get("_sequence", {}).get("status")
    return {"ok": "approved", "flag": "NEEDS WORK"}.get(status, "review started")


def rewrite_image_paths(seq_dir, old_city, new_city, continent):
    """Keep the informative image_path fields of the GT in line with a renamed folder ('/' or Windows '\\')."""
    path = Path(seq_dir) / "ground_truth_labels.jsonl"
    pattern = re.compile(rf"([\\/]){re.escape(continent)}([\\/]){re.escape(old_city)}([\\/])")
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in records:
        if isinstance(r.get("image_path"), str):
            r["image_path"] = pattern.sub(rf"\g<1>{continent}\g<2>{new_city}\g<3>", r["image_path"])
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def remove_empty_parents(path, stop):
    path, stop = Path(path), Path(stop).resolve()
    while path.resolve() != stop and path.is_dir() and not any(path.iterdir()):
        path.rmdir()
        path = path.parent


def import_submissions(root, author=None, dry_run=False):
    root = Path(root)
    submissions = find_submissions(root)
    if not submissions:
        sys.exit(f"No sequences (folders with ground_truth_labels.jsonl) under {root}")

    existing = {seq: image_ids(seq) for seq in dataset_sequences()}
    stop = root.resolve()
    while stop.name != "students_data" and stop != stop.parent and "students_data" in stop.parts:
        stop = stop.parent                       # clean up empty folders up to students_data/ itself
    taken, plan = set(), []

    for seq in submissions:
        continent, city = seq.parts[-2], seq.parts[-1]
        who = author or infer_author(seq)
        row = {"src": seq, "continent": continent, "city": city, "author": who, "review": review_status(seq)}
        if continent not in CONTINENTS:
            row["skip"] = f"unknown continent '{continent}' (expected one of {', '.join(CONTINENTS)})"
        elif not who:
            row["skip"] = "author unknown: not under students_data/<Author>/, use --author"
        else:
            ids = image_ids(seq)
            same = next((e for e, e_ids in existing.items() if ids and ids & e_ids), None)
            if same:
                row["skip"] = f"same images already in the dataset as {same.relative_to(DATA_DIR)}"
            else:
                row["dest"] = free_name(continent, city, taken)
                taken.add((continent, row["dest"]))
        plan.append(row)

    print(f"{'Author':8s} {'Sequence':28s} {'-> data/':30s} Review")
    for r in plan:
        target = f"SKIP: {r['skip']}" if "skip" in r else f"{r['continent']}/{r['dest']}" + \
                 (f"  (renamed from {r['city']})" if r["dest"] != r["city"] else "")
        print(f"{str(r['author']):8s} {r['continent'] + '/' + r['city']:28s} {target:30s} {r['review'] if 'skip' not in r else ''}")
    todo = [r for r in plan if "skip" not in r]
    print(f"\n{len(todo)} to import, {len(plan) - len(todo)} skipped.")
    if dry_run or not todo:
        if dry_run:
            print("Dry run: nothing was moved.")
        return

    today = date.today().isoformat()
    for r in todo:
        src, dest = r["src"], DATA_DIR / r["continent"] / r["dest"]
        info = build_info(src, author=r["author"], added_to_dataset=today, in_paper=False)
        if r["dest"] != r["city"]:
            info["original_name"] = r["city"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        save_info(dest, info)
        (dest / "source.txt").unlink(missing_ok=True)       # folded into info.json
        if r["dest"] != r["city"]:
            rewrite_image_paths(dest, r["city"], r["dest"], r["continent"])
        remove_empty_parents(src.parent, stop)
        print(f"Imported {r['continent']}/{r['dest']} ({r['author']})")
    print(f"\nDone. Run the models on them with: python vqa/run.py   (sequences with existing results are re-run too;"
          f" use --continent/--city to limit)")


def tag_existing(author, in_paper):
    tagged = 0
    for seq in dataset_sequences():
        if load_info(seq):
            continue
        # paper sequences are annotated AND were evaluated (have model results)
        paper = bool(in_paper and (seq / "ground_truth_labels.jsonl").exists() and (seq / "results").is_dir())
        save_info(seq, build_info(seq, author=author, in_paper=paper))
        tagged += 1
        print(f"Tagged {seq.relative_to(DATA_DIR)} -> {author}" + (" (in paper)" if paper else ""))
    print(f"\n{tagged} sequences tagged ({INFO_FILE}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import students' sequences into data/ and record their author")
    parser.add_argument("path", nargs="?", help="students_data, or one student's folder (students_data/<Author>)")
    parser.add_argument("--author", help="Author name (default: the folder below students_data/)")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be imported")
    parser.add_argument("--tag-existing", metavar="AUTHOR",
                        help="Write info.json with this author for sequences already in data/ that have none")
    parser.add_argument("--in-paper", action="store_true",
                        help="With --tag-existing: mark the annotated sequences with model results as part of the paper's dataset")
    args = parser.parse_args()

    if args.tag_existing:
        tag_existing(args.tag_existing, args.in_paper)
    elif args.path:
        import_submissions(args.path, args.author, args.dry_run)
    else:
        parser.error("give a path to import (e.g. students_data) or --tag-existing AUTHOR")
