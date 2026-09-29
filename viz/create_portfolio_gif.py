#!/usr/bin/env python3
"""Generate a portfolio GIF showcasing VQA for pedestrian safety on event maps."""

import json
import string
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from io import BytesIO

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path("/home/tvallsc/Documents/VQA-for-event-maps/data")
CONTINENT   = "America"
CITY        = "BuenosAires"
MODEL       = "qwen-vl"
PROMPTS_JSON = Path("/home/tvallsc/Documents/VQA-for-event-maps/inout/vqa_prompts.json")
OUTPUT_GIF  = Path("/home/tvallsc/Documents/VQA-for-event-maps/viz/portfolio.gif")

CITY_DIR    = DATA_ROOT / CONTINENT / CITY
IMAGE_DIR   = CITY_DIR / "images_selected"
ANSWERS_PATH = CITY_DIR / "results" / MODEL / "answers.jsonl"
GPS_CSV     = CITY_DIR / "gps_positions.csv"

# Selected images: 3 high-risk + 1 low-risk for contrast
SELECTED = [
    "0008_799960560949841.jpg",   # 7 yes
    "0016_462093951732599.jpg",   # 5 yes
    "0120_936854640448321.jpg",   # 5 yes
    "0024_498326984631782.jpg",   # 1 yes (safe)
]

# ── Palette ───────────────────────────────────────────────────────────────────
BG          = (14, 17, 29)
PANEL       = (22, 27, 46)
BORDER      = (45, 55, 90)
ACCENT      = (80, 160, 255)
YES_COL     = (220, 75,  75)
NO_COL      = (55,  195, 115)
TEXT_LT     = (210, 215, 230)
TEXT_DIM    = (100, 110, 140)
WHITE       = (255, 255, 255)

FRAME_W, FRAME_H = 1160, 600
IMG_AREA_W  = 530
PANEL_X     = IMG_AREA_W + 20
PANEL_W     = FRAME_W - PANEL_X - 10

FONT_TITLE  = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG    = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_MONO   = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# ── Risk helpers (mirrors viz_utils.py) ───────────────────────────────────────
HAZARD_CONFIG = {
    "CRITICAL": {"weight": 1.0, "ids": ["q_construction_visible", "q_surface_hazardous", "q_pedestrian_not_on_sidewalk"]},
    "HIGH":     {"weight": 0.6, "ids": ["q_crossing_nearby", "q_stairs_visible", "q_obstacle_blocking"]},
    "LOW":      {"weight": 0.3, "ids": ["q_pedestrians_present", "q_vehicle_nearby"]},
}
WEIGHT_LOOKUP = {qid: cfg["weight"] for cfg in HAZARD_CONFIG.values() for qid in cfg["ids"]}
SAFETY_REWARD = 1 / 8

def get_risk(img_qs, primary_ids):
    max_score = sum(WEIGHT_LOOKUP.get(q, 0) for q in primary_ids)
    if max_score <= 0:
        return 0.0
    net = 0.0
    for q_id in primary_ids:
        if q_id not in img_qs:
            continue
        ans = img_qs[q_id].get("answer", "").lower().strip().translate(str.maketrans("", "", string.punctuation))
        w = WEIGHT_LOOKUP.get(q_id, 0)
        if any(p in ans for p in ["yes", "true", "hazard"]):
            net += w
        elif any(p in ans for p in ["no", "false", "safe"]):
            net -= w * SAFETY_REWARD
    return min(max(0, net) / max_score, 1.0)

def risk_label(score):
    if score <= 0.15: return "Very Safe",   (55,  195, 115)
    if score < 0.4:  return "Caution",     (240, 190, 60)
    if score < 0.7:  return "Danger",      (230, 120, 30)
    return                   "High Risk",  (220, 60,  60)

def score_to_hex(score):
    if score <= 0.15: return "#2ecc71"
    if score < 0.4:  return "#f1c40f"
    if score < 0.7:  return "#e67e22"
    return                   "#e74c3c"

# ── Font loaders ──────────────────────────────────────────────────────────────
def fnt(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

# ── Draw helpers ──────────────────────────────────────────────────────────────
def rounded_paste(canvas, img, xy, radius=12):
    """Paste img onto canvas at xy with rounded corners."""
    mask = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius=radius, fill=255)
    canvas.paste(img, xy, mask)

def draw_badge(draw, x, y, w, h, text, bg, fg=WHITE, font=None, radius=6):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=bg)
    tw = draw.textlength(text, font=font)
    draw.text((x + (w - tw) / 2, y + (h - (font.size if font else 12)) / 2), text, fill=fg, font=font)

def pill(draw, x, y, text, color, font, pad_x=10, pad_h=6):
    tw = draw.textlength(text, font=font)
    w = int(tw + pad_x * 2)
    h = font.size + pad_h * 2
    draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=color)
    draw.text((x + pad_x, y + pad_h), text, fill=WHITE, font=font)
    return w, h

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    with open(PROMPTS_JSON) as f:
        cfg = json.load(f)
    primary_ids = cfg["prompt_presets"]["level_1_only"]["enabled_questions"]
    short_labels = {}
    for cat in cfg["prompt_categories"].values():
        for q in cat["questions"]:
            short_labels[q["id"]] = q.get("short_label", q["id"])

    answers = []
    with open(ANSWERS_PATH) as f:
        for line in f:
            answers.append(json.loads(line))

    by_image = defaultdict(dict)
    for a in answers:
        by_image[a["image_name"]][a["question_id"]] = a

    return primary_ids, short_labels, by_image

# ── Frame builders ────────────────────────────────────────────────────────────
def base_canvas():
    img = Image.new("RGB", (FRAME_W, FRAME_H), BG)
    d = ImageDraw.Draw(img)
    # Subtle gradient-like header bar
    d.rectangle([0, 0, FRAME_W, 38], fill=(20, 25, 45))
    f_title = fnt(FONT_TITLE, 14)
    d.text((14, 11), "VQA for Event Maps", fill=ACCENT, font=f_title)
    d.text((FRAME_W - 200, 11), "Buenos Aires, Argentina", fill=TEXT_DIM, font=fnt(FONT_REG, 13))
    return img, d

def make_title_frame():
    img = Image.new("RGB", (FRAME_W, FRAME_H), BG)
    d = ImageDraw.Draw(img)

    # Center block
    cx, cy = FRAME_W // 2, FRAME_H // 2
    f_big   = fnt(FONT_TITLE, 36)
    f_sub   = fnt(FONT_REG,   18)
    f_small = fnt(FONT_REG,   14)

    title = "VQA for Event Maps"
    tw = d.textlength(title, font=f_big)
    d.text((cx - tw / 2, cy - 80), title, fill=WHITE, font=f_big)

    sub = "Pedestrian Safety Detection via Visual Question Answering"
    sw = d.textlength(sub, font=f_sub)
    d.text((cx - sw / 2, cy - 28), sub, fill=TEXT_DIM, font=f_sub)

    # Colored pills for hazard tiers
    labels = [("CRITICAL", YES_COL), ("HIGH", (230,120,30)), ("LOW", (240,190,60)), ("SAFE", NO_COL)]
    pill_f = fnt(FONT_TITLE, 13)
    total_w = sum(int(d.textlength(t, font=pill_f) + 30) for t, _ in labels) + 12 * 3
    px = cx - total_w // 2
    py = cy + 28
    for t, c in labels:
        pw, ph = pill(d, px, py, t, c, pill_f, pad_x=15)
        px += pw + 12

    model_txt = "Model: Qwen-VL"
    mw = d.textlength(model_txt, font=f_small)
    d.text((cx - mw / 2, cy + 80), model_txt, fill=ACCENT, font=f_small)

    return np.array(img)


def make_image_frame(img_name, primary_ids, short_labels, by_image, n_questions_shown):
    """Render one frame: street photo + questions panel."""
    canvas, _ = base_canvas()
    d = ImageDraw.Draw(canvas)

    # ── Left: street photo ────────────────────────────────────────────────────
    try:
        photo = Image.open(IMAGE_DIR / img_name).convert("RGB")
    except Exception:
        photo = Image.new("RGB", (400, 300), (50, 50, 70))

    # Fit photo into allocated area with padding
    ph_max_w = IMG_AREA_W - 20
    ph_max_h = FRAME_H - 60
    photo.thumbnail((ph_max_w, ph_max_h), Image.LANCZOS)
    ph_x = 10
    ph_y = 48 + (ph_max_h - photo.height) // 2
    rounded_paste(canvas, photo, (ph_x, ph_y), radius=10)

    # ── Right panel ───────────────────────────────────────────────────────────
    rx = PANEL_X
    ry = 46
    rw = PANEL_W
    rh = FRAME_H - ry - 8

    d.rounded_rectangle([rx, ry, rx + rw, ry + rh], radius=10, fill=PANEL, outline=BORDER, width=1)

    f_label  = fnt(FONT_TITLE, 12)
    f_answer = fnt(FONT_TITLE, 12)
    f_q      = fnt(FONT_REG,   12)
    f_score  = fnt(FONT_TITLE, 26)
    f_cat    = fnt(FONT_REG,   13)
    f_head   = fnt(FONT_TITLE, 14)

    # Section header
    d.text((rx + 14, ry + 12), "Hazard Questions", fill=ACCENT, font=f_head)
    d.text((rx + 14, ry + 30), f"Image: {img_name.split('_')[0]}", fill=TEXT_DIM, font=fnt(FONT_REG, 11))

    # Question rows
    img_qs = by_image.get(img_name, {})
    row_h = 32
    q_start_y = ry + 54
    q_x = rx + 10

    # Filter to only primary (level 1) questions that exist in answers
    l1_qs = [(qid, short_labels.get(qid, qid)) for qid in primary_ids if qid in img_qs]

    for i, (qid, label) in enumerate(l1_qs):
        if i >= n_questions_shown:
            break
        ans_raw = img_qs[qid].get("answer", "?").strip()
        is_yes  = "yes" in ans_raw.lower()
        ans_txt = "YES" if is_yes else "NO"
        ans_col = YES_COL if is_yes else NO_COL
        dot_col = YES_COL if is_yes else NO_COL

        qy = q_start_y + i * row_h
        # Dot indicator
        d.ellipse([q_x, qy + 9, q_x + 10, qy + 19], fill=dot_col)
        # Label
        d.text((q_x + 16, qy + 7), label, fill=TEXT_LT, font=f_q)
        # Answer badge
        badge_w = 36
        bx = rx + rw - badge_w - 12
        d.rounded_rectangle([bx, qy + 5, bx + badge_w, qy + 23], radius=4, fill=ans_col + (200,) if False else ans_col)
        atw = d.textlength(ans_txt, font=f_answer)
        d.text((bx + (badge_w - atw) / 2, qy + 6), ans_txt, fill=WHITE, font=f_answer)

    # ── Risk score (shown after all questions) ────────────────────────────────
    if n_questions_shown >= len(l1_qs):
        score = get_risk(img_qs, primary_ids)
        label_txt, label_col = risk_label(score)

        sep_y = q_start_y + len(l1_qs) * row_h + 10
        d.line([rx + 10, sep_y, rx + rw - 10, sep_y], fill=BORDER, width=1)

        # Score number
        score_txt = f"{score:.2f}"
        stw = d.textlength(score_txt, font=f_score)
        d.text((rx + rw // 2 - stw // 2, sep_y + 12), score_txt, fill=label_col, font=f_score)

        # Category pill
        cat_txt = label_txt.upper()
        ctw = int(d.textlength(cat_txt, font=f_label) + 24)
        cx2 = rx + rw // 2 - ctw // 2
        d.rounded_rectangle([cx2, sep_y + 52, cx2 + ctw, sep_y + 72], radius=10, fill=label_col)
        ttw = d.textlength(cat_txt, font=f_label)
        d.text((cx2 + (ctw - ttw) / 2, sep_y + 56), cat_txt, fill=WHITE, font=f_label)

        # Bar
        bar_x, bar_y = rx + 14, sep_y + 82
        bar_w, bar_h = rw - 28, 8
        d.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], radius=4, fill=BORDER)
        fill_w = max(8, int(bar_w * score))
        d.rounded_rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], radius=4, fill=label_col)

    return np.array(canvas)


def make_map_frame():
    """Render a static matplotlib risk map of Buenos Aires."""
    gps = pd.read_csv(GPS_CSV).sort_values("filename")

    # Load answers for all images
    answers = []
    with open(ANSWERS_PATH) as f:
        for line in f:
            answers.append(json.loads(line))
    by_image = defaultdict(dict)
    for a in answers:
        by_image[a["image_name"]][a["question_id"]] = a

    with open(PROMPTS_JSON) as f:
        cfg = json.load(f)
    primary_ids = cfg["prompt_presets"]["level_1_only"]["enabled_questions"]

    scores = []
    for _, row in gps.iterrows():
        fname = row["filename"]
        qs = by_image.get(fname, {})
        scores.append(get_risk(qs, primary_ids) if qs else None)
    gps["risk"] = scores

    answered = gps[gps["risk"].notna()].copy()
    unanswered = gps[gps["risk"].isna()].copy()

    fig, ax = plt.subplots(figsize=(FRAME_W / 130, FRAME_H / 130), dpi=130)
    fig.patch.set_facecolor("#0E1120")
    ax.set_facecolor("#0E1120")

    if not unanswered.empty:
        ax.scatter(unanswered["longitude"], unanswered["latitude"],
                   c="#3a3a5a", s=18, zorder=2, linewidths=0)

    # GPS path
    ax.plot(gps["longitude"], gps["latitude"],
            color="#3498db", linewidth=1.2, alpha=0.5, zorder=3, linestyle="--")

    # Risk scatter
    colors = [score_to_hex(s) for s in answered["risk"]]
    sc = ax.scatter(answered["longitude"], answered["latitude"],
                    c=colors, s=55, zorder=5, linewidths=0.6, edgecolors="white", alpha=0.95)

    # Add padding around GPS track for a less cramped view
    lon_pad = (gps["longitude"].max() - gps["longitude"].min()) * 0.25
    lat_pad = (gps["latitude"].max() - gps["latitude"].min()) * 0.25
    ax.set_xlim(gps["longitude"].min() - lon_pad, gps["longitude"].max() + lon_pad)
    ax.set_ylim(gps["latitude"].min() - lat_pad, gps["latitude"].max() + lat_pad)

    # Basemap
    try:
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.DarkMatter, zoom=15)
    except Exception:
        pass

    ax.set_axis_off()

    # Legend
    legend_items = [
        mpatches.Patch(color="#2ecc71", label="Very Safe  (≤0.15)"),
        mpatches.Patch(color="#f1c40f", label="Caution    (0.15–0.4)"),
        mpatches.Patch(color="#e67e22", label="Danger     (0.4–0.7)"),
        mpatches.Patch(color="#e74c3c", label="High Risk  (>0.7)"),
    ]
    leg = ax.legend(handles=legend_items, loc="lower left",
                    framealpha=0.85, facecolor="#16182E", edgecolor="#2d3766",
                    labelcolor="white", fontsize=9, title="Risk Level",
                    title_fontsize=10)
    leg.get_title().set_color("#80A0FF")

    ax.set_title("Pedestrian Risk Event Map  ·  Qwen-VL  ·  Buenos Aires",
                 color="white", fontsize=12, pad=8)

    plt.tight_layout(pad=0.4)
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    map_img = Image.open(buf).convert("RGB").resize((FRAME_W, FRAME_H), Image.LANCZOS)
    return np.array(map_img)


# ── GIF assembly ──────────────────────────────────────────────────────────────
def build_gif():
    primary_ids, short_labels, by_image = load_data()
    frames = []

    TITLE_DUR   = 100   # ms per frame
    IMG_DUR     = 90
    Q_DUR       = 120
    SCORE_DUR   = 140
    HOLD_DUR    = 100
    MAP_DUR     = 120

    # ── Title card ────────────────────────────────────────────────────────────
    title_frame = make_title_frame()
    for _ in range(18):   # ~1.8 s
        frames.append((title_frame, TITLE_DUR))

    # ── Per-image animation ───────────────────────────────────────────────────
    for img_name in SELECTED:
        img_qs   = by_image.get(img_name, {})
        l1_qs    = [qid for qid in primary_ids if qid in img_qs]
        n_total  = len(l1_qs)

        # 2 frames: image appears (no questions yet)
        for _ in range(2):
            frames.append((make_image_frame(img_name, primary_ids, short_labels, by_image, 0), IMG_DUR))

        # Questions reveal one-by-one
        for n in range(1, n_total + 1):
            frames.append((make_image_frame(img_name, primary_ids, short_labels, by_image, n), Q_DUR))

        # Score reveal: 3 frames
        for _ in range(3):
            frames.append((make_image_frame(img_name, primary_ids, short_labels, by_image, n_total + 1), SCORE_DUR))

        # Hold
        for _ in range(8):
            frames.append((make_image_frame(img_name, primary_ids, short_labels, by_image, n_total + 1), HOLD_DUR))

    # ── Risk map ──────────────────────────────────────────────────────────────
    print("Generating risk map (fetching tiles)…")
    map_frame = make_map_frame()
    for _ in range(30):   # ~3.6 s
        frames.append((map_frame, MAP_DUR))

    # ── Write GIF ─────────────────────────────────────────────────────────────
    print(f"Writing {len(frames)} frames → {OUTPUT_GIF}")
    imgs   = [f[0] for f in frames]
    durs   = [f[1] / 1000.0 for f in frames]  # imageio uses seconds

    # Convert to uint8 PIL for palette quantization
    pil_frames = [Image.fromarray(arr).quantize(colors=256, method=Image.Quantize.MEDIANCUT) for arr in imgs]

    pil_frames[0].save(
        OUTPUT_GIF,
        save_all=True,
        append_images=pil_frames[1:],
        duration=[int(d * 1000) for d in durs],
        loop=0,
        optimize=False,
    )
    print(f"Done! Saved to {OUTPUT_GIF}  ({OUTPUT_GIF.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build_gif()
