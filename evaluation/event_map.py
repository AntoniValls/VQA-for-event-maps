"""
Interactive risk event maps (paper §III-D / Fig. 4): keyframes are snapped to the nearest
OpenStreetMap street segment, each segment is coloured by the max risk of its images.

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
import networkx as nx
import osmnx as ox
import pandas as pd
import requests
import xyzservices.providers as xyz
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.paths import PROMPTS_PATH, get_setting, sequence_dir
from evaluation.risk import image_risk, risk_color

# Cache the OpenStreetMap street network downloads in <repo>/cache (git-ignored), whatever the working directory
ox.settings.cache_folder = str(ROOT / "cache")
CACHE_DIR = ROOT / "cache"


def download_street_network(min_lat, min_lon, max_lat, max_lon, pad_m=80):
    """
    Street network covering a GPS bbox (+ a small padding), pulled straight from the main OSM
    0.6 API (api.openstreetmap.org/api/0.6/map) instead of an Overpass query. Every public
    Overpass mirror (overpass-api.de, kumi.systems/private.coffee, monicz.dev, osm.ch, ...) is
    either IP-blocked or regional-only on this network, while the core OSM API isn't - this
    avoids Overpass entirely. Unlike Overpass, this endpoint returns *everything* in the bbox
    (buildings, POIs, not just streets) and is capped at 50k nodes, so the box has to stay
    tight - a fixed 1km radius blows through that cap in any dense city center. We only need
    it to cover the sequence's own GPS points, not an arbitrary radius.
    """
    center_lat = (min_lat + max_lat) / 2
    lat_pad = pad_m / 111_320
    lon_pad = pad_m / (111_320 * math.cos(math.radians(center_lat)))
    bbox = (min_lon - lon_pad, min_lat - lat_pad, max_lon + lon_pad, max_lat + lat_pad)

    CACHE_DIR.mkdir(exist_ok=True)
    xml_path = CACHE_DIR / f"osm_map_{min_lat:.5f}_{min_lon:.5f}_{max_lat:.5f}_{max_lon:.5f}.xml"
    if not xml_path.exists():
        resp = requests.get(
            "https://api.openstreetmap.org/api/0.6/map",
            params={"bbox": ",".join(f"{v:.6f}" for v in bbox)},
            headers={"User-Agent": "VQA-for-event-maps research project (github.com)"},
            timeout=60,
        )
        resp.raise_for_status()
        xml_path.write_bytes(resp.content)

    # retain_all=True: graph_from_xml's default keeps only the largest connected component of
    # the *raw* graph, before any street/building filtering - in a dense area the building-
    # outline mesh can easily outsize the real (thinner) street network, silently discarding
    # every actual street. Keep everything here; _filter_to_walkable takes the largest component
    # of the streets-only graph instead, once buildings etc. are already gone.
    # simplify=False: simplify_graph has a quirk on some of these small/looped bboxes where it
    # drops the 'highway' edge attribute entirely if run before filtering.
    G = ox.graph_from_xml(xml_path, bidirectional=True, simplify=False, retain_all=True)
    G = _filter_to_walkable(G)
    return ox.simplification.simplify_graph(G)


# graph_from_xml has no network-type filter built in - an Overpass query only ever returns
# highway-tagged ways, but the OSM 0.6 API's bbox dump returns *every* way in the box, including
# building footprints and landuse polygons. graph_from_xml turns all of them into graph edges
# regardless, so without this filter most "streets" in the graph are actually building outlines
# (no 'highway' tag at all) - which is also why they never get a GPS point snapped to them.
#
# This keeps footway/path/pedestrian/steps/service/cycleway/track - unlike osmnx's network_type
# ='drive' filter, which drops those - because this is a pedestrian dataset: plazas and
# pedestrian-only streets are exactly where many sequences walk, and a 'drive'-only filter
# leaves those areas with zero street edges at all.
_NON_WALKABLE_HIGHWAY = {
    "abandoned", "bus_guideway", "construction", "corridor", "elevator", "escalator",
    "no", "planned", "platform", "proposed", "raceway", "razed", "rest_area",
}


def _filter_to_walkable(G):
    drop = [
        (u, v, k) for u, v, k, data in G.edges(keys=True, data=True)
        if data.get("highway") is None
        or set(data["highway"] if isinstance(data["highway"], list) else [data["highway"]]) & _NON_WALKABLE_HIGHWAY
    ]
    G.remove_edges_from(drop)
    G.remove_nodes_from(list(nx.isolates(G)))
    return ox.truncate.largest_component(G, strongly=False)


# Background map styles. OpenStreetMap's own tile servers block maps opened as local files,
# so the OSM-based backgrounds come from MapTiler, which needs a (free) key.
TILE_STYLES = ["streets", "positron", "light", "backdrop", "blank"]
MAPTILER_STYLES = {"streets": "Streets", "positron": "Positron", "backdrop": "Backdrop"}

def add_background(m, style):
    """
    streets:  MapTiler Streets (OpenStreetMap data)         -> needs MAPTILER_API_KEY in .env
    positron: MapTiler Positron, light grey, has labels     -> needs MAPTILER_API_KEY in .env
    backdrop: MapTiler Backdrop, plain grey - still has street-name labels at close zoom
              -> needs MAPTILER_API_KEY in .env
    light:    Esri light grey canvas, no key needed (native zoom <= 16, less detail)
    blank:    no tile imagery at all - solid flat grey, guaranteed no labels of any kind.
              The simplest option; the drawn street graph itself is the only geography shown.

    Every raster style still has some native color (parks, water, ...), so the tile pane is
    also forced to true greyscale via CSS - this only touches the basemap, not the risk-colored
    segments/markers drawn on top (those live in a different Leaflet pane).
    """
    m.get_root().html.add_child(folium.Element(
        "<style>.leaflet-tile-pane { filter: grayscale(100%); }</style>"))

    if style == "blank":
        m.get_root().html.add_child(folium.Element(
            "<style>.leaflet-container { background: #dcdcdc !important; }</style>"))
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


def generate_waypoints_only_map(waypoints, output_html_path, map_title, tiles="streets"):
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
                       tiles="streets"):
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

   # 3. OSM Extraction
    center_lat, center_lon = gps_df_raw['latitude'].mean(), gps_df_raw['longitude'].mean()
    G = download_street_network(
        gps_df_raw['latitude'].min(), gps_df_raw['longitude'].min(),
        gps_df_raw['latitude'].max(), gps_df_raw['longitude'].max(),
    )
    
    # Colour every edge on the shortest path *between* each consecutive pair of images, not just
    # each image's single nearest edge - near a multi-way intersection, the globally-nearest edge
    # to one isolated point can be a street the path never actually used, while the shortest path
    # between its neighbours is guaranteed to follow the real street network and connect up.
    gps_df_filtered = gps_df_filtered.sort_values('filename').reset_index(drop=True)
    gps_df_filtered['node_id'] = ox.distance.nearest_nodes(G, gps_df_filtered['longitude'], gps_df_filtered['latitude'])

    edge_risks = {}
    for i in range(len(gps_df_filtered) - 1):
        row_a, row_b = gps_df_filtered.iloc[i], gps_df_filtered.iloc[i + 1]
        risk = max(image_risk(answers_by_image.get(row_a['filename'], {})),
                   image_risk(answers_by_image.get(row_b['filename'], {})))
        # int(): a pandas int64 column always hands back numpy.int64 scalars, whose str() differs
        # from G's own plain-int node ids ("np.int64(123)" vs "123") - left as-is, the (u, v, k)
        # edge_risks keys below would never match a real edge's.
        try:
            path_nodes = nx.shortest_path(G, int(row_a['node_id']), int(row_b['node_id']), weight='length')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            k = min(G[u][v], key=lambda kk: G[u][v][kk].get('length', 0))
            eid = str((u, v, k))
            # Take the maximum risk if multiple image-pairs' paths share a segment
            if eid not in edge_risks or risk > edge_risks[eid]:
                edge_risks[eid] = risk

    # The very first/last point has only one neighbour to route to/from, so routing alone only
    # covers the edge *up to* its nearest node - the stretch of street past that node, right
    # where a mid-block endpoint actually sits (nearest_nodes snaps to the closest intersection,
    # which can legitimately be 30+m away), never gets coloured. Every interior point is
    # bracketed by two neighbours so doesn't have this gap - only the two endpoints need the
    # direct nearest-edge fallback (using it on every point would bring back the original bug:
    # an isolated point's single nearest edge can be the wrong street near a busy intersection).
    if len(gps_df_filtered) >= 1:
        endpoints = gps_df_filtered.iloc[[0, -1]]
        own_edges = ox.distance.nearest_edges(G, endpoints['longitude'], endpoints['latitude'])
        for (u, v, k), (_, row) in zip(own_edges, endpoints.iterrows()):
            eid = str((u, v, k))
            risk = image_risk(answers_by_image.get(row['filename'], {}))
            if eid not in edge_risks or risk > edge_risks[eid]:
                edge_risks[eid] = risk

    # 4. Initialize Map
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

    # 5. Draw Road segments with Grey fallback
    for u, v, k, edge_data in G.edges(keys=True, data=True):
        eid_str = str((u, v, k))
        
        # Coordinates logic
        if 'geometry' in edge_data:
            coords = [[lat, lon] for lon, lat in edge_data['geometry'].coords]
        else:
            coords = [[G.nodes[u]['y'], G.nodes[u]['x']], [G.nodes[v]['y'], G.nodes[v]['x']]]
        
        # Check if this segment exists in our risk-mapped dictionary
        if eid_str in edge_risks:
            line_color = risk_color(edge_risks[eid_str])
            line_weight = 12
            line_opacity = 0.7
            tooltip_txt = f"Analyzed Risk: {edge_risks[eid_str]:.2f}"
        else:
            # Segment was not matched to any answered image
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

    # 6. Draw Path (Thin blue sequence)
    folium.PolyLine(gps_df_raw[['latitude', 'longitude']].values.tolist(), color='#3498db', weight=2, opacity=0.8, dash_array='5, 5').add_to(m)

    # 7. Add Markers with Full Context
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
    legend_html = f'''
    <div style="position: fixed; bottom: 30px; left: 100px; width: 240px; 
    background-color: white; border:3px solid #7f8c8d; z-index:9999; font-size:16px;
    padding: 15px; border-radius: 12px; font-family: 'Segoe UI', Arial, sans-serif; 
    box-shadow: 4px 4px 15px rgba(0,0,0,0.3); line-height: 1.6;">
    <b style="font-size: 18px; border-bottom: 1px solid #ccc; display: block; margin-bottom: 10px;">Safety Legend</b>
    
    <div style="margin-bottom: 5px;">
        <i class="fa fa-square" style="color:{risk_color(0.0)}; font-size:22px; vertical-align: middle;"></i> 
        <span style="margin-left: 10px;">Safe</span>
    </div>
    <div style="margin-bottom: 5px;">
        <i class="fa fa-square" style="color:{risk_color(0.3)}; font-size:22px; vertical-align: middle;"></i> 
        <span style="margin-left: 10px;">Caution</span>
    </div>
    <div style="margin-bottom: 5px;">
        <i class="fa fa-square" style="color:{risk_color(0.6)}; font-size:22px; vertical-align: middle;"></i> 
        <span style="margin-left: 10px;">Danger</span>
    </div>
    <div style="margin-bottom: 5px;">
        <i class="fa fa-square" style="color:{risk_color(0.9)}; font-size:22px; vertical-align: middle;"></i> 
        <span style="margin-left: 10px;">High Risk</span>
    </div>
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
    parser.add_argument("--tiles", choices=TILE_STYLES, default="streets",
                        help="Background map: streets (default) or positron (paper look), both need MAPTILER_API_KEY; "
                             "light (no key)")
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
