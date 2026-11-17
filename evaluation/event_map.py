"""
Interactive risk event maps (paper §III-D / Fig. 4): the walked GPS track is split into fixed-
length segments, each coloured by the max risk of whichever analyzed keyframes fall inside it.

    python evaluation/event_map.py --continent America --city NewYork --model qwen-vl
"""
import argparse
import base64
import json
import math
import os
import sys
import webbrowser
from io import BytesIO
from pathlib import Path

import folium
import pandas as pd
import xyzservices.providers as xyz
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.paths import PROMPTS_PATH, get_setting, sequence_dir
from evaluation.risk import RISK_CATEGORIES, image_risk, risk_color

# Length of each coloured track segment. Earlier versions snapped keyframes onto an OpenStreetMap
# street graph instead, but that kept hitting edge cases (buildings parsed as streets, the wrong
# street coloured near a multi-way intersection, endpoints with no edge coverage, ...). Colouring
# fixed-length chunks of the actual walked track sidesteps all of that - no street graph involved.
SEGMENT_LENGTH_M = 50


def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between two (lat, lon) points."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def segment_track(gps_df_raw, gps_df_filtered, answers_by_image, segment_length_m=SEGMENT_LENGTH_M):
    """
    Split the walked GPS track (gps_df_raw, in order) into contiguous ~segment_length_m chunks,
    each coloured by the max risk of the analyzed images (gps_df_filtered) that fall inside it.

    Returns a list of (coords, risk_or_None) in walking order - coords is the [lat, lon] pairs
    making up that chunk (to draw as one polyline), risk_or_None is None where no analyzed image
    falls in that stretch.
    """
    points = gps_df_raw[['latitude', 'longitude']].values.tolist()
    cum_dist = [0.0]
    for (lat1, lon1), (lat2, lon2) in zip(points[:-1], points[1:]):
        cum_dist.append(cum_dist[-1] + _haversine_m(lat1, lon1, lat2, lon2))
    dist_by_filename = dict(zip(gps_df_raw['filename'], cum_dist))

    segment_risks = {}
    for _, row in gps_df_filtered.iterrows():
        seg_idx = int(dist_by_filename[row['filename']] // segment_length_m)
        risk = image_risk(answers_by_image.get(row['filename'], {}))
        if seg_idx not in segment_risks or risk > segment_risks[seg_idx]:
            segment_risks[seg_idx] = risk

    chunks = []
    current_idx = int(cum_dist[0] // segment_length_m)
    current_coords = [points[0]]
    for i in range(1, len(points)):
        seg_idx = int(cum_dist[i] // segment_length_m)
        current_coords.append(points[i])
        if seg_idx != current_idx:
            chunks.append((current_coords, segment_risks.get(current_idx)))
            current_coords = [points[i]]  # repeat the boundary point so segments connect visually
            current_idx = seg_idx
    chunks.append((current_coords, segment_risks.get(current_idx)))
    return chunks


# Background map styles. OpenStreetMap's own tile servers block maps opened as local files,
# so the OSM-based backgrounds come from MapTiler, which needs a (free) key.
TILE_STYLES = ["backdrop", "streets", "positron", "light", "blank"]
MAPTILER_STYLES = {"streets": "Streets", "positron": "Positron", "backdrop": "Backdrop"}

def add_background(m, style):
    """
    backdrop: MapTiler Backdrop, plain grey streets+buildings  -> needs MAPTILER_API_KEY in .env
    streets:  MapTiler Streets (OpenStreetMap data)            -> needs MAPTILER_API_KEY in .env
    positron: MapTiler Positron, light grey                    -> needs MAPTILER_API_KEY in .env
    light:    Esri light grey canvas, no key needed (native zoom <= 16, less detail)
    blank:    solid flat grey, no imagery at all - not even street/building shapes, just the
              drawn track and markers on their own.

    All of these are raster tiles with any labels baked directly into the pixels (street/shop
    names, at close zoom), which there's no way to selectively strip via CSS - that needs vector
    tiles and a different renderer (MapLibre GL instead of Leaflet), which this doesn't use.
    CartoDB's once-free "no labels" raster tiles now require a paid account and silently return
    an "API KEY REQUIRED" placeholder tile without one, so that option was removed rather than
    left in as a trap.

    Every raster style still has some native color (parks, water, ...), so the tile pane is
    also forced to true greyscale via CSS - this only touches the basemap, not the risk-colored
    segments/markers drawn on top (those live in a different Leaflet pane).
    """
    m.get_root().html.add_child(folium.Element(
        "<style>.leaflet-tile-pane { filter: grayscale(100%); }</style>"))

    if style == "blank":
        # A real (if trivial) tile layer, not just a CSS background colour on the map container:
        # Leaflet only finishes initializing its panes - which the markers/polylines drawn on top
        # are positioned relative to - once a tile layer actually loads, so a map with no layer at
        # all can end up with nothing visibly rendered. This repeats one solid-grey 1x1 PNG as every
        # tile, so the "no imagery" look is identical but Leaflet still goes through its normal init.
        folium.TileLayer(
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGO4c+cOAAUsApUYk4oFAAAAAElFTkSuQmCC",
            attr=" ", name="blank", max_zoom=21,
        ).add_to(m)
        return

    if style in MAPTILER_STYLES:
        key = get_setting("MAPTILER_API_KEY")
        if not key or key.startswith("your_"):
            raise RuntimeError(
                f"--tiles {style} needs MAPTILER_API_KEY in .env (free account: https://cloud.maptiler.com/account/keys/). "
                "Without a key, use --tiles light.")
        provider = getattr(xyz.MapTiler, MAPTILER_STYLES[style])(key=key)
        folium.TileLayer(provider, name=style, max_zoom=21).add_to(m)
    elif style == "light":
        folium.TileLayer(
            "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ", name="light",
            max_native_zoom=16, max_zoom=19).add_to(m)
    else:
        raise ValueError(f"Unknown tile style '{style}' (choose from {TILE_STYLES})")


def load_positions(seq_dir):
    """Per-image positions (filename, latitude, longitude) from the sequence metadata.json."""
    metadata_path = Path(seq_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json in {seq_dir} (needed for the GPS positions)")
    with open(metadata_path) as f:
        df = pd.DataFrame(json.load(f))
    df = df.dropna(subset=["latitude", "longitude"])
    return df.sort_values(by="filename")[["filename", "latitude", "longitude"]]


# Barcelona Biel-Glasses sequences (Europe/00-22) have no metadata.json - only Europe/08 was ever
# reconstructed into per-frame GPS. The rest only have this: the recording app's raw absolute
# waypoints per sequence (IRI_00..IRI_22), with no frame-level correspondence. A full per-frame
# reconstruction (VIO + OrienterNet heading + OSM boundary correction) is a separate, bigger task -
# for now these sequences just draw the real recorded path from its waypoints, nothing inferred.
BARCELONA_GT_PATH = Path.home() / "Documents" / "PCPP-IRI" / "data" / "BielGlasses" / "IRI_sequences_GT.txt"


def load_barcelona_waypoints(city):
    """Raw (lat, lon) waypoints for Europe/<city> from IRI_sequences_GT.txt, or None if this
    isn't one of those sequences (missing file, or no IRI_<city> block in it)."""
    if not BARCELONA_GT_PATH.exists():
        return None
    text = BARCELONA_GT_PATH.read_text(encoding="utf-8-sig")
    for block in text.strip().split("\n\n"):
        lines = [l for l in block.strip().splitlines() if l.strip()]
        points, name = [], None
        for i, line in enumerate(lines):
            if i == 0:
                continue  # header row ("type  latitude  longitude  name  desc")
            parts = line.split("\t")
            if len(parts) > 3 and parts[3].strip():
                name = parts[3].strip()
            points.append((float(parts[1]), float(parts[2])))
        if name == f"IRI_{city}":
            return points
    return None


def generate_waypoints_only_map(waypoints, output_html_path, map_title, tiles="positron"):
    """Minimal map for a Barcelona sequence with no per-frame GPS: just the recorded path and
    its waypoints, no risk colouring (there's no per-image position to colour)."""
    center_lat = sum(lat for lat, _ in waypoints) / len(waypoints)
    center_lon = sum(lon for _, lon in waypoints) / len(waypoints)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles=None)
    add_background(m, tiles)

    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
    background: white; padding: 8px 16px; border-radius: 8px; font-size: 15px;
    font-family: sans-serif; font-weight: bold; z-index: 9999;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.2);">
    {map_title} - recorded path only (no per-frame GPS/risk for this sequence)
    </div>'''
    m.get_root().html.add_child(folium.Element(title_html))

    folium.PolyLine(waypoints, color='#3498db', weight=3, opacity=0.8, dash_array='5, 5').add_to(m)
    for i, (lat, lon) in enumerate(waypoints):
        folium.CircleMarker(
            [lat, lon], radius=5, color='white', weight=1,
            fill=True, fill_color='#3498db', fill_opacity=1,
            tooltip=f"Waypoint {i + 1}/{len(waypoints)}",
        ).add_to(m)

    m.save(output_html_path)


def generate_event_map(seq_dir,
                       answers_jsonl_path,
                       output_html_path,
                       map_title,
                       prompt_json_path=PROMPTS_PATH,
                       show=True,
                       tiles="positron"):
    """
    Args:
        seq_dir: data/<Continent>/<City> (metadata.json and images_selected/ are read from here)
        answers_jsonl_path: model answers.jsonl or ground_truth_labels.jsonl
        map_title: title shown on the map (model name or "Ground Truth")
        tiles: background style, one of TILE_STYLES
    """
    image_dir = Path(seq_dir) / "images_selected"

    if not (Path(seq_dir) / "metadata.json").exists():
        waypoints = load_barcelona_waypoints(Path(seq_dir).name)
        if waypoints:
            generate_waypoints_only_map(waypoints, output_html_path, map_title, tiles)
            if show:
                webbrowser.open('file://' + os.path.abspath(output_html_path))
            return

    # 1. Load Config
    question_full_text, question_to_category, category_names = {}, {}, {}
    if os.path.exists(prompt_json_path):
        with open(prompt_json_path, 'r') as f:
            prompt_config = json.load(f)
            for cat_id, cat_data in prompt_config['prompt_categories'].items():
                category_names[cat_id] = cat_data.get('name', cat_id)
                for q in cat_data['questions']:
                    question_full_text[q['id']] = q['text']
                    question_to_category[q['id']] = cat_id

    # 2. Load Data
    gps_df_raw = load_positions(seq_dir)

    with open(answers_jsonl_path, 'r') as f:
        answers_data = [json.loads(line) for line in f if line.strip()]
    
    # Organize answers by image — same logic works for both GT and model outputs
    answers_by_image = {}
    for ans in answers_data:
        img_name = ans['image_name']
        if img_name not in answers_by_image:
            answers_by_image[img_name] = {}
        answers_by_image[img_name][ans['question_id']] = ans

    # CRITICAL STEP: Filter GPS data to ONLY include images that have answers
    answered_filenames = set(answers_by_image.keys())
    gps_df_filtered = gps_df_raw[gps_df_raw['filename'].isin(answered_filenames)].copy()

    center_lat, center_lon = gps_df_raw['latitude'].mean(), gps_df_raw['longitude'].mean()
    gps_df_filtered = gps_df_filtered.sort_values('filename').reset_index(drop=True)

    # 3. Initialize Map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles=None)
    add_background(m, tiles)

    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
    background: white; padding: 8px 16px; border-radius: 8px; font-size: 15px;
    font-family: sans-serif; font-weight: bold; z-index: 9999;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.2);">
    {map_title}
    </div>'''
    m.get_root().html.add_child(folium.Element(title_html))

    # 4. Draw the track as ~100m segments, each coloured by the max risk of the images in it
    for coords, risk in segment_track(gps_df_raw, gps_df_filtered, answers_by_image):
        if risk is not None:
            line_color = risk_color(risk)
            line_weight = 16
            line_opacity = 0.7
            tooltip_txt = f"Analyzed Risk: {risk:.2f}"
        else:
            line_color = "#bdc3c7"  # Grey
            line_weight = 4         # Thinner for background streets
            line_opacity = 0.3
            tooltip_txt = "No Image Data"

        folium.PolyLine(
            coords,
            color=line_color,
            weight=line_weight,
            opacity=line_opacity,
            tooltip=tooltip_txt
        ).add_to(m)

    # 5. Draw Path (Thin blue sequence)
    folium.PolyLine(gps_df_raw[['latitude', 'longitude']].values.tolist(), color='#3498db', weight=2, opacity=0.8, dash_array='5, 5').add_to(m)

    # 6. Add Markers with Full Context
    for img_name, img_questions in answers_by_image.items():
        gps_row = gps_df_filtered[gps_df_filtered['filename'] == img_name]
        if gps_row.empty: continue
        lat, lon = gps_row.iloc[0]['latitude'], gps_row.iloc[0]['longitude']
        
        # Base64 Image
        img_html = ""
        try:
            with Image.open(image_dir / img_name) as img:
                img.thumbnail((300, 200))
                if img.mode != "RGB":
                    img = img.convert("RGB")  # JPEG can't encode RGBA/P/... (e.g. the ZED PNG keyframes)
                buf = BytesIO(); img.save(buf, format="JPEG")
                img_str = base64.b64encode(buf.getvalue()).decode()
                img_html = f'<img src="data:image/jpeg;base64,{img_str}" style="width:100%; border-radius:5px; margin-bottom:5px;">'
        except OSError: pass

        # Categorized popup logic
        cat_tables = {}
        for q_id, ans in img_questions.items():
            cid = question_to_category.get(q_id, 'Other')
            if cid not in cat_tables: cat_tables[cid] = []
            val = ans.get('answer', 'N/A')
            color = 'red' if 'yes' in val.lower() else 'green' if 'no' in val.lower() else 'black'
            cat_tables[cid].append(f'<tr><td style="font-size:10px;">{question_full_text.get(q_id, q_id)}</td><td style="color:{color}; font-weight:bold;">{val}</td></tr>')

        popup_html = f'<div style="width:350px; max-height:350px; overflow-y:auto; font-family:sans-serif;">{img_html}'
        for cid in sorted(cat_tables.keys(), key=lambda x: 'level_1' not in x):
            popup_html += f'<div style="margin-top:8px; border-top:1px solid #ddd;"><b>{category_names.get(cid, cid)}</b><table style="width:100%;">{"".join(cat_tables[cid])}</table></div>'
        popup_html += '</div>'

        norm_score = image_risk(img_questions)
        folium.CircleMarker(
            [lat, lon], radius=6, 
            color='white', weight=1,
            fill=True, fill_color=risk_color(norm_score), fill_opacity=1,
            popup=folium.Popup(popup_html)
        ).add_to(m)

    # 8. Updated Legend (Enlarged and Font-Optimized)
    category_rows = "".join(
        f'<div style="margin-bottom: 5px;">'
        f'<i class="fa fa-square" style="color:{color}; font-size:22px; vertical-align: middle;"></i> '
        f'<span style="margin-left: 10px;">{name}</span>'
        f'</div>'
        for name, color in RISK_CATEGORIES
    )
    legend_html = f'''
    <div style="position: fixed; bottom: 30px; left: 100px; width: 240px;
    background-color: white; border:3px solid #7f8c8d; z-index:9999; font-size:16px;
    padding: 15px; border-radius: 12px; font-family: 'Segoe UI', Arial, sans-serif;
    box-shadow: 4px 4px 15px rgba(0,0,0,0.3); line-height: 1.6;">
    <b style="font-size: 18px; border-bottom: 1px solid #ccc; display: block; margin-bottom: 10px;">Safety Legend</b>

    {category_rows}
    <div style="margin-bottom: 15px;">
        <i class="fa fa-square" style="color:#bdc3c7; font-size:22px; vertical-align: middle;"></i> 
        <span style="margin-left: 10px; color: #7f8c8d;">No Data</span>
    </div>

    <b style="font-size: 16px; display: block; margin-top: 10px; border-top: 1px solid #eee; padding-top: 10px;">Navigation</b>
    <div style="margin-top: 5px;">
        <span style="display: inline-block; width: 30px; border-bottom: 4px dashed #3498db; vertical-align: middle; margin-bottom: 6px;"></span>
        <span style="margin-left: 10px;">GPS Path</span>
    </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save(output_html_path)
    if show: webbrowser.open('file://' + os.path.abspath(output_html_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the interactive risk event maps (model + GT)")
    parser.add_argument("--continent", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--model", default=None, help="Model name (must match a results/<model>/answers.jsonl)")
    parser.add_argument("--tiles", choices=TILE_STYLES, default="positron",
                        help="Background map: positron (default, needs MAPTILER_API_KEY, street names "
                             "visible at close zoom - raster tiles, can't selectively strip labels), "
                             "streets/backdrop (also need the key), light (no key, Esri canvas), "
                             "blank (no imagery at all)")
    parser.add_argument("--no-show", action="store_true", help="Don't open the maps in the browser")
    args = parser.parse_args()

    seq_dir = sequence_dir(args.continent, args.city)
    if args.model:
        model_dir = seq_dir / "results" / args.model

    try:
        add_background(folium.Map(tiles=None), args.tiles)   # fail fast if the map key is missing

        # Model map
        if args.model:
            generate_event_map(seq_dir, model_dir / "answers.jsonl", str(model_dir / "interactive_map.html"),
                               map_title=args.model, show=not args.no_show, tiles=args.tiles)

        # GT map — saved in the sequence folder for easy comparison
        generate_event_map(seq_dir, seq_dir / "ground_truth_labels.jsonl", str(seq_dir / "interactive_map_gt.html"),
                           map_title="Ground Truth", show=not args.no_show, tiles=args.tiles)
    except (RuntimeError, FileNotFoundError) as e:
        sys.exit(f"ERROR: {e}")
