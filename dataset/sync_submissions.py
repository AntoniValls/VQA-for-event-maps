"""
End-to-end sync of external submissions (Jorge, Loe, ...): pulls new archives from each
contributor's own Google Drive, imports the labelled sequences into data/ (keyframes + ground
truth + metadata only, the full-resolution images/ dump is dropped - it's reconstructable from
Mapillary via info.json/metadata.json if ever needed), and runs the VQA models on exactly the
newly imported sequences. Submissions are imported as-is, with no automatic QA gate (use
review_sequences.py separately if you ever want to check one by hand).

    python dataset/sync_submissions.py                      # pull, import, run models, done
    python dataset/sync_submissions.py --backup              # ...then also back up all of data/ to Proton
    python dataset/sync_submissions.py --dry-run             # show what would be imported, touch nothing
    python dataset/sync_submissions.py --skip-run            # import only, run models yourself later
    python dataset/sync_submissions.py --models qwen-vl       # only run this model on the new sequences

Dedup/renaming on import (handled by import_submissions.py, unchanged by this script):
  - A sequence whose Mapillary images are already in data/ (same capture re-submitted) is not
    re-imported, but is still checked against --models and run on any it's missing results for
    (e.g. a model added after it was first imported).
  - A sequence whose folder name collides with an existing one but has different images gets the
    next free number instead (data/Asia/Singapore -> data/Asia/Singapore2).

One-time setup, per contributor remote:
  1. rclone config -> n) New remote, type "drive". rclone's shared client_id is being retired
     during 2026, so make your own first (free, ~5 min): a Google Cloud project with the Drive API
     enabled, an OAuth consent screen (External, add yourself as a test user), and an OAuth client
     ID of type "Desktop app" -> https://rclone.org/drive/#making-your-own-client-id. Then:
       rclone config update <name> client_id "..." client_secret "..."
  2. If the contributor shared a folder directly with your Google account, it shows up under
     `rclone lsd <name>: --drive-shared-with-me` and you can set that permanently:
       rclone config update <name> shared_with_me true
     If instead they sent you a "anyone with the link" sharing URL, it won't appear under
     shared_with_me at all - pin the remote's root straight to that folder's ID (the
     .../folders/<ID>?usp=sharing part of the link) instead:
       rclone config update <name> shared_with_me false root_folder_id "<ID>"
  3. In .env: VQA_DRIVE_REMOTES="Jorge=jorge:,Loe=loe:" (<Author>=<rclone remote>:<path>,
     comma-separated; path is empty when the remote's root is already pinned to the right folder
     as above; the author name is recorded in info.json)
  4. Optional, for --backup to copy all of data/ (already lean: no full-res images, just
     keyframes+GT+metadata+results) to Proton Drive afterwards: set VQA_BACKUP_REMOTE="proton:VQA-Backups"
     in .env, and configure that remote once with `rclone config` -> n) New remote -> type
     "protondrive" (experimental backend) -> your Proton email + password.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.paths import DATA_DIR, get_setting, sequence_dir
from dataset.import_submissions import find_submissions, import_submissions
from vqa.run import run_sequence

STUDENTS_DATA = ROOT / "students_data"
MANIFEST_PATH = STUDENTS_DATA / ".synced.json"
ARCHIVE_SUFFIXES = {".zip", ".tar", ".tar.gz", ".tgz"}
DEFAULT_MODELS = ["qwen-vl", "gemma-4", "llava", "qwen3-vl"]


def parse_drive_remotes(spec):
    """"Jorge=jorge-drive:path,Loe=loe-drive:path" -> {"Jorge": "jorge-drive:path", "Loe": "loe-drive:path"}"""
    remotes = {}
    for entry in filter(None, (e.strip() for e in spec.split(","))):
        author, _, remote = entry.partition("=")
        if not remote:
            sys.exit(f"Bad VQA_DRIVE_REMOTES entry '{entry}', expected <Author>=<remote>:<path>")
        remotes[author.strip()] = remote.strip()
    return remotes


def load_manifest():
    return set(json.loads(MANIFEST_PATH.read_text())) if MANIFEST_PATH.exists() else set()


def save_manifest(seen):
    STUDENTS_DATA.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(sorted(seen), indent=2))


def archive_identity(path):
    """Name + size: good enough to tell "already handled this one" apart from a same-named re-upload."""
    return f"{path.relative_to(STUDENTS_DATA)}:{path.stat().st_size}"


def find_archives(root):
    return [p for p in root.rglob("*") if p.is_file() and (
        p.suffix.lower() in ARCHIVE_SUFFIXES or p.name.lower().endswith(".tar.gz"))]


def pull_from_drive(remotes):
    for author, remote in remotes.items():
        dest = STUDENTS_DATA / author
        print(f"Pulling new submissions for {author} from {remote} ...")
        subprocess.run(["rclone", "copy", remote, str(dest), "--progress"], check=True)


def extract_archive(archive_path, keep_archive=False):
    """Extracts next to the archive (same name, no extension). Removes the archive unless keep_archive
    (dry run: nothing should be deleted, so a real run can still process it for real afterwards)."""
    dest = archive_path.with_name(archive_path.name.split(".")[0])
    print(f"Extracting {archive_path.relative_to(STUDENTS_DATA)} -> {dest.relative_to(STUDENTS_DATA)}")
    shutil.unpack_archive(str(archive_path), str(dest))
    if not keep_archive:
        archive_path.unlink()
    return dest


def strip_full_images(seq_dir):
    full = seq_dir / "images"
    if full.is_dir():
        shutil.rmtree(full)


def process_new_archives(dry_run):
    """Extracts every not-yet-seen archive under students_data/ and strips full images/ from every
    sequence found inside. Returns the archives that failed to extract (left in place, untouched, so
    they're retried next run once re-uploaded). import_submissions() does the actual move into data/
    afterwards."""
    seen = load_manifest()
    archives = [p for p in find_archives(STUDENTS_DATA) if archive_identity(p) not in seen]
    if not archives:
        print("No new archives.")
        return []

    failed = []
    for archive in archives:
        identity = archive_identity(archive)
        try:
            extracted = extract_archive(archive, keep_archive=dry_run)
        except Exception as e:
            print(f"ERROR extracting {archive.relative_to(STUDENTS_DATA)}: {e} "
                  f"(looks corrupted/incomplete - left in place, ask the contributor to re-upload it)")
            failed.append(archive.relative_to(STUDENTS_DATA))
            partial = archive.with_name(archive.name.split(".")[0])
            if partial.is_dir():
                shutil.rmtree(partial)
            continue
        for seq_dir in find_submissions(extracted):
            if dry_run:
                print(f"Would import {seq_dir.relative_to(STUDENTS_DATA)} (images/ left in place for this dry run).")
            else:
                strip_full_images(seq_dir)
        if not dry_run:
            seen.add(identity)
            save_manifest(seen)
    return failed


def run_models_on(sequences, models, preset, num_keyframes, overwrite):
    """Runs every model missing results for each sequence (all of them with --overwrite).
    Returns the subset of `sequences` that actually had at least one model run."""
    ran = set()
    for model in models:
        for continent, city in sequences:
            answers_path = sequence_dir(continent, city) / "results" / model / "answers.jsonl"
            if answers_path.exists() and not overwrite:
                continue
            try:
                run_sequence(sequence_dir(continent, city), model, preset, num_keyframes)
                ran.add((continent, city))
            except Exception as e:
                print(f"ERROR running {model} on {continent}/{city}: {e}")
    return ran


def backup_dataset(backup_remote):
    """Copies the whole (already lean) data/ folder to backup_remote. A plain copy, not sync: never
    deletes anything on the remote, even if a sequence is later renamed or removed locally."""
    print(f"Backing up {DATA_DIR} -> {backup_remote}")
    subprocess.run(["rclone", "copy", str(DATA_DIR), backup_remote, "--progress"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--remotes", default=get_setting("VQA_DRIVE_REMOTES"),
                        help="<Author>=<remote>:<path>,... to pull submissions from (default: VQA_DRIVE_REMOTES in .env)")
    parser.add_argument("--backup", action="store_true",
                        help="Back up all of data/ to --backup-remote/VQA_BACKUP_REMOTE after syncing")
    parser.add_argument("--backup-remote", default=get_setting("VQA_BACKUP_REMOTE"),
                        help="rclone remote:path to back up data/ to, used with --backup (default: VQA_BACKUP_REMOTE in .env)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help=f"Models to run (default: {' '.join(DEFAULT_MODELS)})")
    parser.add_argument("--preset", default="full_hierarchical")
    parser.add_argument("--num-keyframes", type=int, default=20)
    parser.add_argument("--skip-run", action="store_true", help="Import only; don't run the VQA models")
    parser.add_argument("--overwrite", action="store_true", help="Re-run models even if answers.jsonl already exists")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen; touch nothing")
    args = parser.parse_args()

    if not args.remotes:
        sys.exit("No Drive remotes configured. Set VQA_DRIVE_REMOTES in .env "
                  "(e.g. \"Jorge=jorge:,Loe=loe:\"), or pass --remotes. "
                  "See the module docstring for one-time rclone setup.")
    remotes = parse_drive_remotes(args.remotes)
    if args.backup and not args.backup_remote:
        sys.exit("--backup needs a destination: set VQA_BACKUP_REMOTE in .env (e.g. \"proton:VQA-Backups\") "
                  "or pass --backup-remote.")

    pull_from_drive(remotes)
    failed = process_new_archives(args.dry_run)

    imported, duplicates = [], []
    if not find_submissions(STUDENTS_DATA):
        print("\nNothing to import.")
    else:
        imported, duplicates = import_submissions(STUDENTS_DATA, dry_run=args.dry_run)

    if not args.dry_run and (imported or duplicates):
        # Resubmissions of a sequence already in data/ (same images) aren't re-imported, but if
        # it's missing results for any of --models (e.g. a model added since it was first
        # imported), those still need to run on it.
        to_check = imported + [d for d in duplicates if d not in imported]
        if not args.skip_run:
            run_models_on(to_check, args.models, args.preset, args.num_keyframes, args.overwrite)

        print(f"\nDone. Imported {len(imported)} sequence(s): {', '.join(f'{c}/{s}' for c, s in imported)}")
        if duplicates:
            print(f"{len(duplicates)} resubmission(s) already in data/ (checked for missing model results): "
                  f"{', '.join(f'{c}/{s}' for c, s in duplicates)}")

    if args.backup and not args.dry_run:
        backup_dataset(args.backup_remote)

    if failed:
        print(f"\n{len(failed)} archive(s) failed to extract and were skipped (not imported, not deleted): "
              f"{', '.join(str(f) for f in failed)}")


if __name__ == "__main__":
    main()
