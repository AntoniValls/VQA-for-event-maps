"""
Interactive risk event maps (paper §III-D / Fig. 4): keyframes are snapped to the nearest
OpenStreetMap street segment, each segment is coloured by the max risk of its images.

    python evaluation/event_map.py --continent America --city NewYork --model qwen-vl
"""
import argparse
import base64
import json
import os
import sys
import webbrowser
from io import BytesIO
from pathlib import Path

import folium
import osmnx as ox
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.paths import PROMPTS_PATH, sequence_dir
from evaluation.risk import image_risk, risk_color


def load_positions(seq_dir):
    """Per-image positions (filename, latitude, longitude) from the sequence metadata.json."""
    metadata_path = Path(seq_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json in {seq_dir} (needed for the GPS positions)")
    with open(metadata_path) as f:
        df = pd.DataFrame(json.load(f))
    df = df.dropna(subset=["latitude", "longitude"])
    return df.sort_values(by="filename")[["filename", "latitude", "longitude"]]


def generate_event_map(seq_dir,
                       answers_jsonl_path,
                       output_html_path,
                       map_title,
                       prompt_json_path=PROMPTS_PATH,
                       show=True):
    """
    Args:
        seq_dir: data/<Continent>/<City> (metadata.json and images_selected/ are read from here)
        answers_jsonl_path: model answers.jsonl or ground_truth_labels.jsonl
        map_title: title shown on the map (model name or "Ground Truth")
    """
    image_dir = Path(seq_dir) / "images_selected"

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
    G = ox.graph_from_point((center_lat, center_lon), dist=1000, network_type='drive')
    
    # Map ONLY filtered images to their single nearest edge
    # This prevents the "Green Ghost" effect from unanalyzed GPS points
    nearest_edges = ox.distance.nearest_edges(G, gps_df_filtered['longitude'], gps_df_filtered['latitude'])
    gps_df_filtered['edge_id'] = [str(edge) for edge in nearest_edges]
    
    # Aggregate Max Risk per walked segment
    edge_risks = {}
    for _, row in gps_df_filtered.iterrows():
        eid = row['edge_id']
        norm_score = image_risk(answers_by_image.get(row['filename'], {}))
        # Take the maximum risk if multiple images map to the same segment
        if eid not in edge_risks or norm_score > edge_risks[eid]:
            edge_risks[eid] = norm_score

    # 4. Initialize Map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles='CartoDB positron')

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
            line_weight = 14
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
    parser.add_argument("--model", default="qwen-vl")
    parser.add_argument("--no-show", action="store_true", help="Don't open the maps in the browser")
    args = parser.parse_args()

    seq_dir = sequence_dir(args.continent, args.city)
    model_dir = seq_dir / "results" / args.model

    # Model map
    generate_event_map(seq_dir, model_dir / "answers.jsonl", str(model_dir / "interactive_map.html"),
                       map_title=args.model, show=not args.no_show)

    # GT map — saved in the sequence folder for easy comparison
    generate_event_map(seq_dir, seq_dir / "ground_truth_labels.jsonl", str(seq_dir / "interactive_map_gt.html"),
                       map_title="Ground Truth", show=not args.no_show)
