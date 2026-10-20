"""
Review newly labelled sequences (e.g. a student's submission) before they join the dataset.

    python dataset/review_sequences.py students_data/Jorge         # then open http://localhost:5001
    python dataset/review_sequences.py data/Asia/Hanoi             # a single sequence works too

For every sequence found under the given folder (any folder holding images/ or ground_truth_labels.jsonl):
  - automatic checks: 20 keyframes, every question answered, follow-ups consistent with the hierarchy,
    duplicates, positions, on_foot, keyframe spread, copied/fast labelling
  - overview: source, GPS track on a map, all frames with the keyframes highlighted, hazards found
  - label editor: the question tree of each keyframe with Yes/No toggles. Every change is saved at once
    (first change: ground_truth_labels.jsonl.bak) and logged in review_log.jsonl
  - review status per image (OK / flagged + note) in review.json, and a feedback report for the student
"""
import argparse
import hashlib
import json
import shutil
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.paths import get_setting
from common.sequence_info import load_info
from evaluation.risk import image_risk, risk_category, RISK_CATEGORIES
from vqa.prompt_manager import PromptManager

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
NUM_KEYFRAMES = 20
MIN_FRAMES = 100                 # README §5.1: aim for sequences of >= 100 images
FAST_LABELLING_SECONDS = 20      # less than this per keyframe is suspicious
WEAK_CATEGORIES = {"q_pedestrian_not_on_sidewalk", "q_construction_visible", "q_stairs_visible", "q_surface_hazardous"}
THUMB_DIR = ROOT / "cache" / "thumbs"

PM = PromptManager(preset="full_hierarchical")
L1_IDS = [q['id'] for q in PM.get_initial_prompts()]

app = Flask(__name__)
SEQUENCES = []                   # list of Path, filled in __main__
REVIEW_ROOT = None


# ----------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------

def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

def discover_sequences(root):
    root = Path(root)
    found = set()
    for marker in ["ground_truth_labels.jsonl", "images"]:
        for p in root.rglob(marker):
            if p.parent.name != "results":
                found.add(p.parent)
    if (root / "images").is_dir() or (root / "ground_truth_labels.jsonl").exists():
        found.add(root)
    return sorted(found)

def seq_name(seq_dir):
    """<Continent>/<City> (the last two folders)."""
    return "/".join(Path(seq_dir).parts[-2:])

def gt_path(seq_dir):
    return Path(seq_dir) / "ground_truth_labels.jsonl"

def load_gt(seq_dir):
    path = gt_path(seq_dir)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

def save_gt(seq_dir, records):
    path = gt_path(seq_dir)
    backup = path.with_suffix(".jsonl.bak")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def log_change(seq_dir, entry):
    entry = {"time": datetime.now().isoformat(timespec="seconds"), **entry}
    with open(Path(seq_dir) / "review_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def load_log(seq_dir):
    path = Path(seq_dir) / "review_log.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

def load_review(seq_dir):
    path = Path(seq_dir) / "review.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

def save_review(seq_dir, review):
    (Path(seq_dir) / "review.json").write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")

def load_metadata(seq_dir):
    path = Path(seq_dir) / "metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def answers_by_image(records):
    by_img = {}
    for r in records:
        by_img.setdefault(r["image_name"], {})[r["question_id"]] = r
    return by_img

def norm(answer):
    return str(answer).strip().lower()

def children(qid):
    return [c for c in PM.dependency_map.get(qid, []) if c in PM.all_prompts]

def descendants(qid):
    out = []
    for c in children(qid):
        out.append(c)
        out.extend(descendants(c))
    return out


# ----------------------------------------------------------------------------
# Automatic checks
# ----------------------------------------------------------------------------

def check_sequence(seq_dir):
    """Returns (checks, stats). checks: list of {"level": error|warning|info, "msg", "image"?}"""
    seq_dir = Path(seq_dir)
    checks = []
    def add(level, msg, image=None):
        checks.append({"level": level, "msg": msg, "image": image})

    images = list_images(seq_dir / "images")
    selected = list_images(seq_dir / "images_selected")
    records = load_gt(seq_dir)
    by_img = answers_by_image(records)
    metadata = load_metadata(seq_dir)

    # --- frames and keyframes
    if not images:
        add("error", "No images/ folder or it is empty")
    elif len(images) < MIN_FRAMES:
        add("warning", f"Only {len(images)} frames downloaded (README asks for sequences of about {MIN_FRAMES}+)")
    if len(selected) != NUM_KEYFRAMES:
        add("error", f"images_selected/ has {len(selected)} images, expected {NUM_KEYFRAMES}")
    missing_src = [s for s in selected if s not in images]
    if images and missing_src:
        add("warning", f"{len(missing_src)} keyframes are not in images/ (renamed or from another sequence?)")

    if images and len(images) >= NUM_KEYFRAMES and not missing_src:
        stride = len(images) // NUM_KEYFRAMES
        segments = {}
        for s in selected:
            seg = min(images.index(s) // stride, NUM_KEYFRAMES - 1)
            segments.setdefault(seg, []).append(s)
        empty = [i + 1 for i in range(NUM_KEYFRAMES) if i not in segments]
        crowded = [v for v in segments.values() if len(v) > 1]
        if empty or crowded:
            add("warning", f"Keyframes not spread one per segment: empty segments {empty or '-'}, "
                           f"{sum(len(v) for v in crowded)} keyframes share a segment")

    # --- labels
    if not records:
        add("error", "No ground_truth_labels.jsonl (sequence not labelled yet)")
    seen = {}
    for r in records:
        key = (r.get("image_name"), r.get("question_id"))
        seen[key] = seen.get(key, 0) + 1
    dups = [k for k, n in seen.items() if n > 1]
    if dups:
        add("error", f"{len(dups)} duplicated answers (same image and question), e.g. {dups[0][0]} / {dups[0][1]}")

    for img in by_img:
        if img not in selected:
            add("error", f"Labels for {img}, which is not in images_selected/", img)

    level_mismatch = 0
    for img in selected:
        ans = by_img.get(img, {})
        if not ans:
            add("error", "Keyframe has no labels", img)
            continue
        missing_l1 = [q for q in L1_IDS if q not in ans]
        if missing_l1:
            add("error", f"{len(missing_l1)} Level-1 questions unanswered: {', '.join(missing_l1)}", img)
        for qid, r in ans.items():
            if qid not in PM.all_prompts:
                add("error", f"Unknown question id {qid}", img)
                continue
            if norm(r.get("answer")) not in ("yes", "no"):
                add("error", f"{qid}: answer is '{r.get('answer')}', not yes/no", img)
            parent = PM.all_prompts[qid].get("dependency")
            if parent and norm(ans.get(parent, {}).get("answer")) != "yes":
                add("error", f"{qid} answered although its parent {parent} is not 'yes' (orphan)", img)
            if r.get("level") != PM.get_level(qid):
                level_mismatch += 1
            if norm(r.get("answer")) == "yes":
                missing_children = [c for c in children(qid) if c not in ans]
                if missing_children:
                    add("error", f"{qid} is 'yes' but follow-ups are unanswered: {', '.join(missing_children)}", img)
    if level_mismatch:
        add("info", f"{level_mismatch} answers have a wrong 'level' field (harmless: the evaluation uses the hierarchy)")

    # --- copied answers (R key) and labelling speed
    prev = None
    for img in selected:
        cur = {q: norm(r.get("answer")) for q, r in by_img.get(img, {}).items()}
        if prev is not None and cur and cur == prev:
            add("info", "All answers identical to the previous keyframe (R key?): check it", img)
        prev = cur
    stamps = []
    for img in selected:
        ts = [r.get("timestamp") for r in by_img.get(img, {}).values() if r.get("timestamp")]
        if ts:
            stamps.append(min(datetime.fromisoformat(t) for t in ts))
    if len(stamps) > 2:
        stamps.sort()
        gaps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
        gaps = [g for g in gaps if g < 1800]                          # ignore breaks
        if gaps:
            median = sorted(gaps)[len(gaps) // 2]
            if median < FAST_LABELLING_SECONDS:
                add("warning", f"Very fast labelling: median {median:.0f} s per keyframe")
            else:
                add("info", f"Labelling pace: median {median:.0f} s per keyframe")

    # --- metadata / positions
    if metadata is None:
        add("warning", "No metadata.json: no positions, the event map can't be drawn")
    else:
        meta_by_name = {m.get("filename"): m for m in metadata}
        no_pos = [s for s in selected if s not in meta_by_name or meta_by_name[s].get("latitude") is None]
        if no_pos:
            add("warning", f"{len(no_pos)} keyframes have no GPS position in metadata.json")
        if any(m.get("is_pano") or m.get("camera_type") in ("spherical", "equirectangular") for m in metadata):
            add("error", "Contains 360° images")
        on_foot = {m.get("on_foot") for m in metadata}
        if False in on_foot:
            add("error", "Mapillary marks this sequence as NOT captured on foot")
        elif None in on_foot:
            add("info", "Mapillary doesn't know whether it was captured on foot: check the point of view")
        original = sum(1 for m in metadata if m.get("position_source") == "original")
        if original:
            add("info", f"{original} of {len(metadata)} images use the raw device GPS (no computed position)")

    # --- stats
    hazards = {q: sum(1 for img in selected if norm(by_img.get(img, {}).get(q, {}).get("answer")) == "yes")
               for q in L1_IDS}
    risks = [image_risk({q: {"answer": r.get("answer")} for q, r in by_img.get(img, {}).items()}) for img in selected]
    stats = {
        "n_images": len(images), "n_selected": len(selected), "n_answers": len(records),
        "hazards": hazards,
        "risk_counts": [sum(1 for r in risks if risk_category(r) == i) for i in range(len(RISK_CATEGORIES))],
        "errors": sum(1 for c in checks if c["level"] == "error"),
        "warnings": sum(1 for c in checks if c["level"] == "warning"),
    }
    return checks, stats


# ----------------------------------------------------------------------------
# Label editing
# ----------------------------------------------------------------------------

def image_tree(seq_dir, img):
    ans = answers_by_image(load_gt(seq_dir)).get(img, {})
    def node(qid):
        q = PM.all_prompts[qid]
        a = norm(ans[qid]["answer"]) if qid in ans else None
        return {"id": qid, "text": q["text"], "label": PM.get_short_label(q), "level": PM.get_level(qid),
                "answer": a, "children": [node(c) for c in children(qid)] if a == "yes" else []}
    tree = [node(q) for q in L1_IDS]
    orphans = [qid for qid in ans if qid in PM.all_prompts and PM.all_prompts[qid].get("dependency")
               and norm(ans.get(PM.all_prompts[qid]["dependency"], {}).get("answer")) != "yes"]
    risk = image_risk({q: {"answer": r.get("answer")} for q, r in ans.items()})
    return {"tree": tree, "orphans": orphans, "risk": risk,
            "risk_label": RISK_CATEGORIES[risk_category(risk)][0], "risk_color": RISK_CATEGORIES[risk_category(risk)][1]}

def set_answer(seq_dir, img, qid, answer):
    if qid not in PM.all_prompts or answer not in ("yes", "no"):
        abort(400, "bad question or answer")
    records = load_gt(seq_dir)
    ans = answers_by_image(records).get(img, {})
    parent = PM.all_prompts[qid].get("dependency")
    if parent and norm(ans.get(parent, {}).get("answer")) != "yes":
        abort(400, f"parent {parent} is not 'yes'")

    old = norm(ans[qid]["answer"]) if qid in ans else None
    removed = []
    now = datetime.now().isoformat()
    if qid in ans:
        for r in records:
            if r["image_name"] == img and r["question_id"] == qid:
                r["answer"], r["timestamp"] = answer, now
    else:
        template = next((r for r in records if r["image_name"] == img), {})
        q = PM.all_prompts[qid]
        new = {"image_path": template.get("image_path", f"images_selected/{img}"), "image_name": img,
               "image_index": template.get("image_index"), "question_id": qid, "question": q["text"],
               "answer": answer, "level": PM.get_level(qid)}
        if parent:
            new["parent_question"] = parent
        new.update({"short_label": PM.get_short_label(q), "timestamp": now})
        last = max((i for i, r in enumerate(records) if r["image_name"] == img), default=len(records) - 1)
        records.insert(last + 1, new)
    if answer == "no":
        drop = set(descendants(qid))
        removed = [r["question_id"] for r in records if r["image_name"] == img and r["question_id"] in drop]
        records = [r for r in records if not (r["image_name"] == img and r["question_id"] in drop)]

    save_gt(seq_dir, records)
    if old != answer:
        log_change(seq_dir, {"image": img, "question_id": qid, "old": old, "new": answer, "removed": removed})

def remove_orphans(seq_dir):
    records = load_gt(seq_dir)
    by_img = answers_by_image(records)
    keep, removed = [], []
    for r in records:
        parent = PM.all_prompts.get(r["question_id"], {}).get("dependency")
        if parent and norm(by_img[r["image_name"]].get(parent, {}).get("answer")) != "yes":
            removed.append((r["image_name"], r["question_id"]))
        else:
            keep.append(r)
    if removed:
        save_gt(seq_dir, keep)
        for img, qid in removed:
            log_change(seq_dir, {"image": img, "question_id": qid, "old": "orphan", "new": None, "removed": []})
    return len(removed)


# ----------------------------------------------------------------------------
# Feedback report
# ----------------------------------------------------------------------------

def build_report():
    lines = [f"# Review of {REVIEW_ROOT}", f"_{datetime.now():%Y-%m-%d %H:%M}_", ""]
    for seq_dir in SEQUENCES:
        checks, stats = check_sequence(seq_dir)
        review = load_review(seq_dir)
        log = load_log(seq_dir)
        seq_review = review.get("_sequence", {})
        lines.append(f"## {seq_name(seq_dir)}")
        if seq_review.get("status"):
            verdict = "APPROVED" if seq_review["status"] == "ok" else "NEEDS WORK"
            lines.append(f"**Sequence: {verdict}**" + (f": {seq_review['note']}" if seq_review.get("note") else ""))
        found = [PM.get_short_label(PM.all_prompts[q]) for q, n in stats["hazards"].items() if n]
        lines.append(f"- {stats['n_images']} frames, {stats['n_selected']} keyframes, {stats['n_answers']} answers. "
                     f"Hazards present: {', '.join(found) or 'none'}")
        problems = [c for c in checks if c["level"] in ("error", "warning")]
        if problems:
            lines.append("- Automatic checks:")
            lines += [f"  - {c['level'].upper()}: {c['msg']}" + (f" ({c['image']})" if c.get("image") else "") for c in problems]
        flagged = [(img, v) for img, v in review.items() if img != "_sequence" and v.get("status") == "flag"]
        if flagged:
            lines.append("- Flagged keyframes:")
            lines += [f"  - {img}: {v.get('note') or '(no note)'}" for img, v in flagged]
        if log:
            lines.append(f"- Corrections made in review ({len(log)}):")
            for e in log:
                label = PM.get_short_label(PM.all_prompts[e["question_id"]]) if e["question_id"] in PM.all_prompts else e["question_id"]
                if e["old"] == "orphan":
                    change = "removed (its parent question was not 'yes')"
                else:
                    change = f"{e['old'] or '(unanswered)'} -> {e['new']}"
                    if e.get("removed"):
                        change += f" (follow-ups removed: {len(e['removed'])})"
                lines.append(f"  - {e['image']}: {label}: {change}")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

def get_seq(sid):
    if not 0 <= sid < len(SEQUENCES):
        abort(404)
    return SEQUENCES[sid]

def page(body_js, data):
    return PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/")).replace("__SCRIPT__", body_js)

@app.route("/")
def index():
    rows = []
    for sid, seq_dir in enumerate(SEQUENCES):
        checks, stats = check_sequence(seq_dir)
        review = load_review(seq_dir)
        reviewed = sum(1 for k, v in review.items() if k != "_sequence" and v.get("status"))
        rows.append({"sid": sid, "name": seq_name(seq_dir), "path": str(seq_dir.relative_to(REVIEW_ROOT) if seq_dir != REVIEW_ROOT else "."),
                     "stats": stats, "reviewed": reviewed, "status": review.get("_sequence", {}).get("status")})
    weak = [{"id": q, "label": PM.get_short_label(PM.all_prompts[q])} for q in L1_IDS]
    return page(INDEX_JS, {"root": str(REVIEW_ROOT), "rows": rows, "l1": weak, "weak": sorted(WEAK_CATEGORIES),
                           "risk": [c[0] for c in RISK_CATEGORIES]})

@app.route("/seq/<int:sid>")
def sequence(sid):
    seq_dir = get_seq(sid)
    checks, stats = check_sequence(seq_dir)
    metadata = load_metadata(seq_dir) or []
    selected = list_images(seq_dir / "images_selected")
    by_img = answers_by_image(load_gt(seq_dir))
    review = load_review(seq_dir)
    keyframes = []
    for i, img in enumerate(selected):
        ans = by_img.get(img, {})
        risk = image_risk({q: {"answer": r.get("answer")} for q, r in ans.items()})
        keyframes.append({"n": i, "name": img, "yes": [q for q in L1_IDS if norm(ans.get(q, {}).get("answer")) == "yes"],
                          "risk": risk, "risk_color": RISK_CATEGORIES[risk_category(risk)][1],
                          "status": review.get(img, {}).get("status"), "note": review.get(img, {}).get("note", "")})
    info = load_info(seq_dir)
    if info:
        source = "\n".join(f"{k}: {v}" for k, v in info.items() if v not in (None, ""))
    else:
        source = (seq_dir / "source.txt").read_text(encoding="utf-8") if (seq_dir / "source.txt").exists() else ""
    track = [[m["latitude"], m["longitude"], m.get("filename")] for m in metadata if m.get("latitude") is not None]
    return page(SEQ_JS, {"sid": sid, "name": seq_name(seq_dir), "path": str(seq_dir), "checks": checks, "stats": stats,
                         "images": list_images(seq_dir / "images"), "keyframes": keyframes, "source": source,
                         "track": track, "tiles": tile_layer(), "seq_review": review.get("_sequence", {}),
                         "l1": {q: PM.get_short_label(PM.all_prompts[q]) for q in L1_IDS},
                         "n_sequences": len(SEQUENCES)})

@app.route("/seq/<int:sid>/kf/<int:n>")
def keyframe(sid, n):
    seq_dir = get_seq(sid)
    selected = list_images(seq_dir / "images_selected")
    if not 0 <= n < len(selected):
        abort(404)
    img = selected[n]
    images = list_images(seq_dir / "images")
    pos = images.index(img) if img in images else None
    context = images[max(0, pos - 3): pos + 4] if pos is not None else []
    checks, _ = check_sequence(seq_dir)
    return page(KF_JS, {"sid": sid, "name": seq_name(seq_dir), "n": n, "total": len(selected), "image": img,
                        "context": context, "frame_pos": pos, "n_frames": len(images),
                        "checks": [c for c in checks if c.get("image") == img],
                        "review": load_review(seq_dir).get(img, {}), **image_tree(seq_dir, img)})

@app.route("/report")
def report():
    return page(REPORT_JS, {"report": build_report()})

@app.route("/file/<int:sid>/<folder>/<name>")
def file(sid, folder, name):
    if folder not in ("images", "images_selected"):
        abort(404)
    return send_from_directory(get_seq(sid) / folder, name)

@app.route("/thumb/<int:sid>/<folder>/<name>")
def thumb(sid, folder, name):
    if folder not in ("images", "images_selected"):
        abort(404)
    src = get_seq(sid) / folder / name
    if not src.exists():
        abort(404)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{src.resolve()}:{src.stat().st_mtime}".encode()).hexdigest()
    dst = THUMB_DIR / f"{key}.jpg"
    if not dst.exists():
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((360, 360))
            im.save(dst, "JPEG", quality=80)
    return send_file(dst, mimetype="image/jpeg")

@app.post("/api/answer")
def api_answer():
    d = request.json
    seq_dir = get_seq(int(d["sid"]))
    set_answer(seq_dir, d["image"], d["question_id"], d["answer"])
    return jsonify(image_tree(seq_dir, d["image"]))

@app.post("/api/review")
def api_review():
    d = request.json
    seq_dir = get_seq(int(d["sid"]))
    review = load_review(seq_dir)
    entry = review.setdefault(d["image"], {})
    for k in ("status", "note"):
        if k in d:
            entry[k] = d[k]
    entry["time"] = datetime.now().isoformat(timespec="seconds")
    save_review(seq_dir, review)
    return jsonify(ok=True)

@app.post("/api/fix_orphans")
def api_fix_orphans():
    return jsonify(removed=remove_orphans(get_seq(int(request.json["sid"]))))

def tile_layer():
    key = get_setting("MAPTILER_API_KEY")
    if key and not key.startswith("your_"):
        return {"url": f"https://api.maptiler.com/maps/streets-v2/{{z}}/{{x}}/{{y}}.png?key={key}",
                "attr": "&copy; MapTiler &copy; OpenStreetMap contributors", "maxNative": 20}
    return {"url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            "attr": "Tiles &copy; Esri", "maxNative": 16}


# ----------------------------------------------------------------------------
# Front-end (plain HTML + JS, no build step)
# ----------------------------------------------------------------------------

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Sequence review</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{--ink:#14213D;--soft:#4A5568;--bg:#F7F6F2;--card:#fff;--line:#DCE1E8;--acc:#B8501F;--err:#C0392B;--warn:#B7791F;--info:#2B5C94;--yes:#B8501F;--no:#2E7D5B}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{background:var(--ink);color:#F7F6F2;padding:12px 24px;display:flex;gap:16px;align-items:center;position:sticky;top:0;z-index:1000}
header a{color:#F2A477;text-decoration:none}main{padding:24px;max-width:1500px;margin:0 auto}
h1{font-size:22px;margin:0}h2{font-size:18px;margin:24px 0 10px}.muted{color:var(--soft)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
table{border-collapse:collapse;width:100%;background:#fff}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;font-size:14px}
th{background:#EAEEF3}tr.click{cursor:pointer}tr.click:hover{background:#FBF3EC}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600}
.e{background:#FADBD8;color:var(--err)}.w{background:#FCEFD5;color:var(--warn)}.i{background:#E3ECF6;color:var(--info)}.ok{background:#D9F0E3;color:var(--no)}.fl{background:#FADBD8;color:var(--err)}
.checks li{margin:4px 0;font-size:14px}.grid{display:grid;gap:12px}
.kf{grid-template-columns:repeat(auto-fill,minmax(210px,1fr))}.kf a{color:inherit;text-decoration:none}
.kf .card{padding:8px}.kf img{width:100%;height:130px;object-fit:cover;border-radius:6px;display:block}
.frames{display:flex;flex-wrap:wrap;gap:4px}.frames img{height:64px;width:86px;object-fit:cover;border-radius:3px;opacity:.55;cursor:pointer;border:3px solid transparent}
.frames img.sel{opacity:1;border-color:var(--acc)}
#map{height:360px;border-radius:10px}
.tree .q{display:flex;gap:10px;align-items:center;padding:6px 8px;border-radius:6px}.tree .q:hover{background:#F2F4F7}
.tree .txt{flex:1;font-size:14px}.tree .lbl{font-weight:600}.sub{margin-left:26px;border-left:2px solid var(--line);padding-left:8px}
.btn{border:1px solid var(--line);background:#fff;border-radius:6px;padding:4px 12px;cursor:pointer;font-weight:600;font-size:13px}
.btn.yes.on{background:var(--yes);color:#fff;border-color:var(--yes)}.btn.no.on{background:var(--no);color:#fff;border-color:var(--no)}
.missing{outline:2px solid var(--err)}.lv{font-size:11px;color:var(--soft);width:22px}
.big{width:100%;max-height:72vh;object-fit:contain;background:#222;border-radius:8px}
.ctx{display:flex;gap:4px;margin-top:6px}.ctx img{height:70px;flex:1;min-width:0;width:0;object-fit:cover;border-radius:4px;opacity:.6;border:3px solid transparent}.ctx img.cur{opacity:1;border-color:var(--acc)}
textarea{width:100%;min-height:70px;font:inherit;padding:8px;border:1px solid var(--line);border-radius:6px}
pre{white-space:pre-wrap;background:#fff;border:1px solid var(--line);padding:16px;border-radius:8px}
kbd{background:#EAEEF3;border-radius:4px;padding:1px 5px;font-size:12px}
.lightbox{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:2000}.lightbox img{max-width:92vw;max-height:92vh}
</style></head><body><header><a href="/"><b>Sequence review</b></a><span id="crumb" class="muted" style="color:#BFC9D9"></span><span style="flex:1"></span><a href="/report">Feedback report</a></header>
<main id="app"></main><div class="lightbox" id="lb" onclick="this.style.display='none'"><img id="lbimg"></div>
<script>const D=__DATA__;const $=s=>document.querySelector(s);const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function post(u,b){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});if(!r.ok){alert(await r.text());throw r}return r.json()}
function pill(l){return {error:'<span class="pill e">error</span>',warning:'<span class="pill w">warning</span>',info:'<span class="pill i">info</span>'}[l]}
function lightbox(src){$('#lbimg').src=src;$('#lb').style.display='flex'}
__SCRIPT__</script></body></html>"""

INDEX_JS = r"""
$('#crumb').textContent=D.root;
const weak=new Set(D.weak);
let h=`<h1>${D.rows.length} sequences in ${esc(D.root)}</h1><p class="muted">Click a sequence to review it. Hazard columns: number of keyframes (of 20) answered <b>yes</b>; the four weak categories of the paper are marked ★.</p>
<table><tr><th>Sequence</th><th>Frames</th><th>Keyframes</th><th>Checks</th><th>Reviewed</th>`+D.l1.map(q=>`<th title="${q.id}">${esc(q.label)}${weak.has(q.id)?' ★':''}</th>`).join('')+`</tr>`;
for(const r of D.rows){const s=r.stats;
 h+=`<tr class="click" onclick="location='/seq/${r.sid}'"><td><b>${esc(r.name)}</b><br><span class="muted">${esc(r.path)}</span></td><td>${s.n_images}</td><td>${s.n_selected}</td>
 <td>${s.errors?`<span class="pill e">${s.errors} error${s.errors>1?'s':''}</span> `:''}${s.warnings?`<span class="pill w">${s.warnings} warning${s.warnings>1?'s':''}</span>`:''}${!s.errors&&!s.warnings?'<span class="pill ok">OK</span>':''}</td>
 <td>${r.reviewed}/${s.n_selected}${r.status?` <span class="pill ${r.status=='ok'?'ok':'fl'}">${r.status=='ok'?'approved':'needs work'}</span>`:''}</td>`+
 D.l1.map(q=>`<td style="${weak.has(q.id)&&s.hazards[q.id]?'font-weight:700;color:#B8501F':''}">${s.hazards[q.id]}</td>`).join('')+`</tr>`}
$('#app').innerHTML=h+'</table>';
"""

SEQ_JS = r"""
$('#crumb').textContent=D.name;
const s=D.stats, sel=new Set(D.keyframes.map(k=>k.name));
const groups=[];for(const c of D.checks){let g=groups.find(g=>g.level==c.level&&g.msg==c.msg);if(!g){g={level:c.level,msg:c.msg,images:[]};groups.push(g)}if(c.image)g.images.push(c.image)}
let h=`<div style="display:flex;gap:12px;align-items:center"><h1>${esc(D.name)}</h1>
 ${D.sid>0?`<a class="btn" href="/seq/${D.sid-1}">← previous sequence</a>`:''}${D.sid<D.n_sequences-1?`<a class="btn" href="/seq/${D.sid+1}">next sequence →</a>`:''}</div>
<p class="muted">${esc(D.path)} · ${s.n_images} frames · ${s.n_selected} keyframes · ${s.n_answers} answers</p>
<div class="grid" style="grid-template-columns:1fr 1fr">
 <div class="card"><h2 style="margin-top:0">Automatic checks</h2><ul class="checks">${groups.length?groups.map(g=>`<li>${pill(g.level)} ${esc(g.msg)}${g.images.length?': '+g.images.map(i=>`<a href="/seq/${D.sid}/kf/${D.keyframes.findIndex(k=>k.name==i)}">#${D.keyframes.findIndex(k=>k.name==i)+1}</a>`).join(', '):''}</li>`).join(''):'<li><span class="pill ok">all checks passed</span></li>'}</ul>
  <button class="btn" onclick="fixOrphans()">Remove orphan answers</button></div>
 <div class="card"><h2 style="margin-top:0">Source</h2><pre style="margin:0;border:none;padding:0">${esc(D.source||'(no info.json or source.txt)').replace(/(https?:\/\/\S+)/g,'<a href="$1" target="_blank">$1</a>')}</pre>
  <h2>Sequence verdict</h2><div style="display:flex;gap:8px;margin-bottom:8px"><button class="btn" id="sok">Approve</button><button class="btn" id="sfl">Needs work</button><span id="sst"></span></div>
  <textarea id="snote" placeholder="Note for the student about the whole sequence">${esc(D.seq_review.note||'')}</textarea></div>
</div>
<h2>Keyframes</h2><div class="grid kf">${D.keyframes.map(k=>`<a href="/seq/${D.sid}/kf/${k.n}"><div class="card" style="border-left:6px solid ${k.risk_color}">
 <img src="/thumb/${D.sid}/images_selected/${encodeURIComponent(k.name)}">
 <div style="font-size:12px;margin-top:6px"><b>#${k.n+1}</b> · risk ${k.risk.toFixed(2)} ${k.status?`<span class="pill ${k.status=='ok'?'ok':'fl'}">${k.status=='ok'?'OK':'flagged'}</span>`:''}</div>
 <div class="muted" style="font-size:12px">${k.yes.map(q=>esc(D.l1[q])).join(', ')||'no hazards'}</div></div></a>`).join('')}</div>
<h2>Route</h2><div id="map"></div>
<h2>All frames <span class="muted" style="font-size:13px;font-weight:400">(keyframes highlighted; click to enlarge)</span></h2>
<div class="frames">${D.images.map(f=>`<img loading="lazy" class="${sel.has(f)?'sel':''}" title="${esc(f)}" src="/thumb/${D.sid}/images/${encodeURIComponent(f)}" onclick="lightbox('/file/${D.sid}/images/${encodeURIComponent(f)}')">`).join('')}</div>`;
$('#app').innerHTML=h;
function showS(st){$('#sst').innerHTML=st?`<span class="pill ${st=='ok'?'ok':'fl'}">${st=='ok'?'approved':'needs work'}</span>`:''}
showS(D.seq_review.status);
const saveS=b=>post('/api/review',{sid:D.sid,image:'_sequence',...b});
$('#sok').onclick=()=>{saveS({status:'ok'});showS('ok')};$('#sfl').onclick=()=>{saveS({status:'flag'});showS('flag')};
$('#snote').onchange=e=>saveS({note:e.target.value});
async function fixOrphans(){const r=await post('/api/fix_orphans',{sid:D.sid});alert(`Removed ${r.removed} orphan answers`);location.reload()}
if(D.track.length){const map=L.map('map');L.tileLayer(D.tiles.url,{attribution:D.tiles.attr,maxNativeZoom:D.tiles.maxNative,maxZoom:20}).addTo(map);
 const line=L.polyline(D.track.map(t=>[t[0],t[1]]),{color:'#2B5C94',weight:3,dashArray:'6 6'}).addTo(map);map.fitBounds(line.getBounds(),{padding:[20,20]});
 for(const k of D.keyframes){const t=D.track.find(t=>t[2]==k.name);if(t)L.circleMarker([t[0],t[1]],{radius:7,color:'#fff',weight:1,fillColor:k.risk_color,fillOpacity:1}).bindTooltip(`#${k.n+1} · risk ${k.risk.toFixed(2)}`).on('click',()=>location=`/seq/${D.sid}/kf/${k.n}`).addTo(map)}}
else $('#map').outerHTML='<p class="muted">No positions in metadata.json</p>';
"""

KF_JS = r"""
$('#crumb').textContent=`${D.name} · keyframe ${D.n+1}/${D.total}`;
const go=n=>{if(n>=0&&n<D.total)location=`/seq/${D.sid}/kf/${n}`};
function renderTree(t){
 const row=q=>`<div class="q ${q.answer?'':'missing'}"><span class="lv">L${q.level}</span><span class="txt"><span class="lbl">${esc(q.label)}</span> · ${esc(q.text)}</span>
  <button class="btn yes ${q.answer=='yes'?'on':''}" onclick="setA('${q.id}','yes')">Yes</button><button class="btn no ${q.answer=='no'?'on':''}" onclick="setA('${q.id}','no')">No</button></div>
  ${q.children.length?`<div class="sub">${q.children.map(row).join('')}</div>`:''}`;
 $('#tree').innerHTML=t.tree.map(row).join('');
 $('#risk').innerHTML=`Risk score <b>${t.risk.toFixed(2)}</b> <span class="pill" style="background:${t.risk_color};color:#14213D">${t.risk_label}</span>`;
 $('#orph').innerHTML=t.orphans.length?`<p>${pill('error')} Orphan answers (parent not yes): ${t.orphans.join(', ')}</p>`:'';
}
async function setA(qid,a){renderTree(await post('/api/answer',{sid:D.sid,image:D.image,question_id:qid,answer:a}))}
function showSt(st){$('#st').innerHTML=st?`<span class="pill ${st=='ok'?'ok':'fl'}">${st=='ok'?'OK':'flagged'}</span>`:''}
const saveR=b=>post('/api/review',{sid:D.sid,image:D.image,...b});
$('#app').innerHTML=`<div style="display:flex;gap:8px;align-items:center;margin-bottom:12px">
 <a class="btn" href="/seq/${D.sid}">↑ sequence</a><button class="btn" onclick="go(${D.n-1})">← prev</button><button class="btn" onclick="go(${D.n+1})">next →</button>
 <b style="margin-left:8px">${esc(D.image)}</b><span class="muted">frame ${D.frame_pos===null?'?':D.frame_pos+1} of ${D.n_frames}</span><span style="flex:1"></span>
 <span class="muted" style="font-size:13px"><kbd>←</kbd><kbd>→</kbd> navigate · <kbd>A</kbd> OK · <kbd>F</kbd> flag</span></div>
<div class="grid" style="grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);align-items:start">
 <div><img class="big" src="/file/${D.sid}/images_selected/${encodeURIComponent(D.image)}" onclick="lightbox(this.src)">
  <div class="ctx">${D.context.map(f=>`<img class="${f==D.image?'cur':''}" title="${esc(f)}" src="/thumb/${D.sid}/images/${encodeURIComponent(f)}" onclick="lightbox('/file/${D.sid}/images/${encodeURIComponent(f)}')">`).join('')}</div>
  <p class="muted" style="font-size:12px">Neighbouring frames of the route (click to enlarge)</p></div>
 <div><div class="card"><div style="display:flex;gap:8px;align-items:center"><button class="btn" id="ok">OK</button><button class="btn" id="fl">Flag</button><span id="st"></span><span style="flex:1"></span><span id="risk"></span></div>
  <textarea id="note" style="margin-top:8px" placeholder="Note for the student about this keyframe">${esc(D.review.note||'')}</textarea>
  ${D.checks.map(c=>`<p style="margin:6px 0;font-size:13px">${pill(c.level)} ${esc(c.msg)}</p>`).join('')}</div>
  <div class="card tree" style="margin-top:12px" id="tree"></div><div id="orph"></div>
  <p class="muted" style="font-size:12px">Every click is saved immediately. Answering No removes that question's follow-ups; answering Yes opens them (red = still unanswered).</p></div>
</div>`;
renderTree(D);showSt(D.review.status);
$('#ok').onclick=()=>{saveR({status:'ok'});showSt('ok')};$('#fl').onclick=()=>{saveR({status:'flag'});showSt('flag')};
$('#note').onchange=e=>saveR({note:e.target.value});
document.addEventListener('keydown',e=>{if(e.target.tagName=='TEXTAREA')return;
 if(e.key=='ArrowLeft')go(D.n-1);if(e.key=='ArrowRight')go(D.n+1);
 if(e.key=='a'||e.key=='A'){saveR({status:'ok'});showSt('ok')}if(e.key=='f'||e.key=='F'){saveR({status:'flag'});showSt('flag');$('#note').focus()}});
"""

REPORT_JS = r"""
$('#app').innerHTML=`<h1>Feedback report</h1><p class="muted">Automatic checks, flagged keyframes with your notes, and the corrections you made. Copy it and send it to the student.</p>
<button class="btn" id="cp">Copy to clipboard</button><pre id="rep"></pre>`;
$('#rep').textContent=D.report;$('#cp').onclick=()=>{navigator.clipboard.writeText(D.report);$('#cp').textContent='Copied'};
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review labelled sequences: checks, overview, label editor")
    parser.add_argument("path", help="A folder with one or more sequences (e.g. students_data/Jorge, or one sequence folder)")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--no-browser", action="store_true", help="Don't open the browser automatically")
    args = parser.parse_args()

    REVIEW_ROOT = Path(args.path).resolve()
    if not REVIEW_ROOT.is_dir():
        sys.exit(f"Not a folder: {REVIEW_ROOT}")
    SEQUENCES = discover_sequences(REVIEW_ROOT)
    if not SEQUENCES:
        sys.exit(f"No sequences (folders with images/ or ground_truth_labels.jsonl) under {REVIEW_ROOT}")

    print(f"Found {len(SEQUENCES)} sequences under {REVIEW_ROOT}:")
    for seq in SEQUENCES:
        _, stats = check_sequence(seq)
        print(f"  {seq_name(seq):28s} {stats['n_selected']:2d} keyframes · {stats['errors']} errors · {stats['warnings']} warnings")
    url = f"http://localhost:{args.port}"
    print(f"\nReview at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    app.run(port=args.port, debug=False)
