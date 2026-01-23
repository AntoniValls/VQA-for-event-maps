from pathlib import Path
import string
import cv2
import json
from PIL import Image
import numpy as np
import pandas as pd 
import folium
from folium import plugins
import base64
from io import BytesIO
import webbrowser
import os

# --- headless-safe preview helper ---
def save_preview(img, out_path="../data/preview.png"):
    import matplotlib
    matplotlib.use("Agg")  # no GUI needed
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    print(f"Saved preview to {out_path}")

def draw_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                              font_scale=0.5, font_thickness=1,
                              text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw text with a background rectangle for better visibility."""
    x, y = position
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Draw background rectangle
    cv2.rectangle(img, (x, y - text_height - 5), (x + text_width, y + baseline), bg_color, -1)
    
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return text_height + baseline + 5

def create_display_frame(image, frame_number, answers_dict, model_name):
    """Create a display frame with the image and answers overlay."""
    # Convert PIL Image to OpenCV format if needed
    if isinstance(image, Image.Image):
        img_display = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img_display = image.copy()
    
    # Add header info
    header_text = f"Frame: {frame_number} | Model: {model_name}"
    draw_text_with_background(img_display, header_text, (10, 30),
                             font_scale=0.7, font_thickness=2,
                             text_color=(0, 255, 0), bg_color=(0, 0, 0))
    
    # Add instruction text
    draw_text_with_background(img_display, "Press any key to continue",
                             (10, img_display.shape[0] - 20),
                             font_scale=0.5, font_thickness=1,
                             text_color=(255, 255, 0), bg_color=(0, 0, 0))
    
    # Add answers
    y_offset = 70
    for prompt_short, answer_data in answers_dict.items():
        answer = answer_data['answer']
        confidence = answer_data.get('confidence')
        
        # Format the text
        if confidence is not None:
            text = f"{prompt_short}: {answer} ({confidence:.2f})"
        else:
            text = f"{prompt_short}: {answer}"
        
        # Choose color based on answer
        if 'yes' in answer.lower() or 'safe' in answer.lower():
            text_color = (0, 255, 0)  # Green
        elif 'no' in answer.lower() or 'wait' in answer.lower() or 'stop' in answer.lower():
            text_color = (0, 0, 255)  # Red
        else:
            text_color = (255, 255, 255)  # White
        
        height = draw_text_with_background(img_display, text, (10, y_offset),
                                          font_scale=0.6, font_thickness=1,
                                          text_color=text_color, bg_color=(0, 0, 0))
        y_offset += height
        
        # Scroll if too many questions
        if y_offset > img_display.shape[0] - 60:
            break
    
    return img_display

import os
import json
import base64
import pandas as pd
import folium
from folium import plugins
from PIL import Image
from io import BytesIO
from pathlib import Path
import webbrowser
import osmnx as ox

def get_normalized_risk(img_questions, primary_ids):
    """
    Calculates a normalized risk score [0, 1] using a weighted penalty/reward system.
    """
    # 1. Configuration: Centralize tiers and penalty ratios
    HAZARD_CONFIG = {
        "CRITICAL": {"weight": 1, "ids": ["q_construction_visible", "q_surface_hazardous", "q_pedestrian_not_on_sidewalk"]},
        "HIGH":     {"weight": 0.6, "ids": ["q_crossing_nearby", "q_stairs_visible", "q_obstacle_blocking"]},
        "LOW":      {"weight": 0.3, "ids": ["q_pedestrians_present", "q_vehicle_nearby"]}
    }
    
    # Ratio for 'No' answers (Reward for safety)
    SAFETY_REWARD_RATIO = 1/6
    
    # 2. Map IDs to weights for O(1) lookup
    weight_lookup = {qid: cfg["weight"] for cfg in HAZARD_CONFIG.values() for qid in cfg["ids"]}
    
    # 3. Calculate Max Possible Score (Denominator)
    # We only consider weights for IDs that are actually in our primary_ids list
    active_weights = [weight_lookup.get(qid, 0) for qid in primary_ids]
    max_theoretical = sum(active_weights)
    
    if max_theoretical <= 0:
        return 0.0

    # 4. Compute Net Score
    net_score = 0.0
    for q_id in primary_ids:
        # Skip if the question wasn't even asked/answered
        if q_id not in img_questions:
            continue
            
        ans = img_questions[q_id].get('answer', '').lower().strip().translate(str.maketrans('', '', string.punctuation))
        weight = weight_lookup.get(q_id, 0)
        
        # Binary classification of the answer
        is_yes = any(pos in ans for pos in ['yes', 'true', 'hazard'])
        is_no = any(neg in ans for neg in ['no', 'false', 'safe'])
        
        if is_yes:
            net_score += weight
        elif is_no:
            net_score -= (weight * SAFETY_REWARD_RATIO)

    # 5. Normalization with Clipping
    # result = (score - min_possible) / (max_possible - min_possible)
    # But since we want 0 to be the "Safe" floor:
    normalized = max(0, net_score) / max_theoretical
    return min(normalized, 1.0)

def get_color_from_normalized(score):
    if score <= 0.15: return '#2ecc71'     # Green (Very Safe)
    if score < 0.4: return '#f1c40f'      # Yellow
    if score < 0.7: return '#e67e22'      # Orange
    return '#e74c3c'                     # Red

def generate_event_map(gps_csv_path, answers_jsonl_path, output_html_path, image_dir, prompt_json_path="../inout/vqa_prompts.json", show=True):
    # 1. Load Config
    primary_ids = []
    question_full_text, question_to_category, category_names = {}, {}, {}
    if os.path.exists(prompt_json_path):
        with open(prompt_json_path, 'r') as f:
            prompt_config = json.load(f)
            primary_ids = prompt_config.get('prompt_presets', {}).get('level_1_only', {}).get('enabled_questions', [])
            for cat_id, cat_data in prompt_config['prompt_categories'].items():
                category_names[cat_id] = cat_data.get('name', cat_id)
                for q in cat_data['questions']:
                    question_full_text[q['id']] = q['text']
                    question_to_category[q['id']] = cat_id

    # 2. Load Data
    gps_df_raw = pd.read_csv(gps_csv_path).sort_values(by='filename')
    with open(answers_jsonl_path, 'r') as f:
        answers_data = [json.loads(line) for line in f]
    
    # Organize answers by image
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
        norm_score = get_normalized_risk(answers_by_image.get(row['filename'], {}), primary_ids)
        # Take the maximum risk if multiple images map to the same segment
        if eid not in edge_risks or norm_score > edge_risks[eid]:
            edge_risks[eid] = norm_score

    # 4. Initialize Map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles='CartoDB positron')

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
            line_color = get_color_from_normalized(edge_risks[eid_str])
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
            with Image.open(Path(image_dir) / img_name) as img:
                img.thumbnail((300, 200))
                buf = BytesIO(); img.save(buf, format="JPEG")
                img_str = base64.b64encode(buf.getvalue()).decode()
                img_html = f'<img src="data:image/jpeg;base64,{img_str}" style="width:100%; border-radius:5px; margin-bottom:5px;">'
        except: pass

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

        norm_score = get_normalized_risk(img_questions, primary_ids)
        folium.CircleMarker(
            [lat, lon], radius=6, 
            color='white', weight=1,
            fill=True, fill_color=get_color_from_normalized(norm_score), fill_opacity=1,
            popup=folium.Popup(popup_html, max_width=350)
        ).add_to(m)

    # 8. Updated Legend for Streets and Path
    legend_html = f'''
    <div style="position: fixed; bottom: 30px; left: 30px; width: 180px; 
    background-color: white; border:2px solid grey; z-index:9999; font-size:12px;
    padding: 12px; border-radius: 8px; font-family: Arial; box-shadow: 2px 2px 5px rgba(0,0,0,0.2);">
    <b style="font-size:13px;">Map Legend</b><hr style="margin:5px 0;">
    <b>Street Risk</b><br>
    <i class="fa fa-minus" style="color:{get_color_from_normalized(0.0)}; font-size:20px;"></i> [0.0] Safe<br>
    <i class="fa fa-minus" style="color:{get_color_from_normalized(0.3)}; font-size:20px;"></i> [0.3] Caution<br>
    <i class="fa fa-minus" style="color:{get_color_from_normalized(0.6)}; font-size:20px;"></i> [0.0] Danger<br>
    <i class="fa fa-minus" style="color:{get_color_from_normalized(0.9)}; font-size:20px;"></i> [0.0] Very Danger<br>
    <i class="fa fa-minus" style="color:#bdc3c7; font-size:20px;"></i> No Image Data<br>
    <br>
    <b>Travel Path</b><br>
    <i class="fa fa-minus" style="color:#3498db; font-size:15px; border-bottom: 2px dashed #3498db;"></i> GPS Sequence
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save(output_html_path)
    if show: webbrowser.open('file://' + os.path.abspath(output_html_path))


if __name__ == "__main__":
    
    CONTINENT = "America"
    CITY = "NewYork"
    MODEL = "qwen-vl"

    gps_csv_path = f"../data/{CONTINENT}/{CITY}/gps_positions.csv"
    answers_path =  os.path.join(Path(gps_csv_path).parent, f"results/{MODEL}/answers.jsonl")
    map_output_path = os.path.join(Path(gps_csv_path).parent, f"results/{MODEL}/interative_map.html")
    image_dir = os.path.join(Path(gps_csv_path).parent, "images_selected")
    generate_event_map(gps_csv_path, answers_path, map_output_path, image_dir, show=True)
