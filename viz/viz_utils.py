from pathlib import Path
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

def generate_event_map(gps_csv_path, answers_jsonl_path, output_html_path, image_dir, prompt_json_path="../inout/vqa_prompts.json", show=True):
    """
    Generate an interactive map with GPS positions and VQA results.
    
    Args:
        gps_csv_path: Path to GPS positions CSV file
        answers_jsonl_path: Path to VQA answers JSONL file
        output_html_path: Path to save the HTML map
        image_dir: Directory containing the images
        prompt_json_path: Path to the prompts JSON file (optional)
    """
    print("\n" + "="*70)
    print("GENERATING INTERACTIVE MAP")
    print("="*70)
    
    # Load GPS data
    gps_df = pd.read_csv(gps_csv_path)
    print(f"Loaded {len(gps_df)} GPS positions")
    
    # Load VQA answers
    answers_data = []
    with open(answers_jsonl_path, 'r') as f:
        for line in f:
            answers_data.append(json.loads(line))
    print(f"Loaded {len(answers_data)} VQA results")
    
    # Load prompt configurations if available
    question_full_text = {}
    if prompt_json_path and os.path.exists(prompt_json_path):
        try:
            with open(prompt_json_path, 'r') as f:
                prompt_config = json.load(f)
            
            # Build mapping of question_id to full question text
            for category_name, category_data in prompt_config['prompt_categories'].items():
                for question in category_data['questions']:
                    question_full_text[question['id']] = {
                        'short_label': question.get('short_label', question['id']),
                        'text': question['text']
                    }
            print(f"Loaded {len(question_full_text)} question definitions")
        except Exception as e:
            print(f"Could not load prompt JSON: {e}")
    
    # Group answers by image and question (to avoid duplicates)
    answers_by_image = {}
    for ans in answers_data:
        img_name = ans['image_name']
        question_id = ans['question_id']
        
        if img_name not in answers_by_image:
            answers_by_image[img_name] = {}
        
        # Only keep the first occurrence of each question for this image
        if question_id not in answers_by_image[img_name]:
            answers_by_image[img_name][question_id] = ans
    
    # Create map centered on the data
    center_lat = gps_df['latitude'].mean()
    center_lon = gps_df['longitude'].mean()
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=16,
        tiles='OpenStreetMap'
    )
    
    # Add alternative tile layers
    folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark').add_to(m)
    
    # Create path line
    path_coordinates = []
    for _, row in gps_df.iterrows():
        path_coordinates.append([row['latitude'], row['longitude']])
    
    # Add path polyline
    folium.PolyLine(
        path_coordinates,
        color='blue',
        weight=3,
        opacity=0.7,
        popup='Path'
    ).add_to(m)
    
    # Add markers for processed images
    for img_name, img_questions in answers_by_image.items():
        # Find GPS position for this image
        gps_row = gps_df[gps_df['filename'] == img_name]
        
        if gps_row.empty:
            continue
        
        lat = gps_row.iloc[0]['latitude']
        lon = gps_row.iloc[0]['longitude']
        
        # Load thumbnail image
        img_path = Path(image_dir) / img_name
        thumbnail_size = (300, 200)
        
        try:
            img = Image.open(img_path)
            img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            
            # Convert image to base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            img_html = f'<img src="data:image/jpeg;base64,{img_str}" style="max-width:300px;"><br>'
        except Exception as e:
            print(f"Error loading thumbnail for {img_name}: {e}")
            img_html = f'<b>{img_name}</b><br>'
        
        # Create popup HTML
        popup_html = f"""
        <div style="width:400px; max-height:600px; overflow-y:auto;">
            {img_html}
            <h4 style="margin:5px 0;">{img_name}</h4>
            <table style="width:100%; font-size:11px; border-collapse: collapse;">
        """
        
        # Add answers to popup (sorted by question_id for consistency)
        for question_id in sorted(img_questions.keys()):
            ans = img_questions[question_id]
            answer = ans['answer']
            confidence = ans.get('confidence')
            
            # Get full question text if available
            if question_id in question_full_text:
                full_question = question_full_text[question_id]['text']
            else:
                full_question = ans.get('question', '')
            
            # Color code based on answer
            if 'yes' in answer.lower() or 'safe' in answer.lower():
                color = 'green'
            elif 'no' in answer.lower() or 'wait' in answer.lower() or 'stop' in answer.lower():
                color = 'red'
            else:
                color = 'black'
            
            conf_str = f" ({confidence:.2f})" if confidence is not None else ""
            
            popup_html += f"""
                <tr style="border-bottom: 1px solid #ddd;">
                    <td colspan="2" style="padding:4px 2px;">
                        <span style="font-size:10px; color:#666;">{full_question}</span>
                    </td>
                    <td style="padding:4px 2px; color:{color};"><b>{answer}</b>{conf_str}</td>
                </tr>
            """
        
        popup_html += """
            </table>
        </div>
        """
        
        # Determine marker color based on answers
        has_danger = any('no' in ans['answer'].lower() or 'wait' in ans['answer'].lower() 
                        for ans in img_questions.values())
        marker_color = 'red' if has_danger else 'green'
        
        # Add marker
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=450),
            tooltip=img_name,
            icon=folium.Icon(color=marker_color, icon='camera', prefix='fa')
        ).add_to(m)
    
    # Add start and end markers
    if not gps_df.empty:
        # Start marker
        folium.Marker(
            location=[gps_df.iloc[0]['latitude'], gps_df.iloc[0]['longitude']],
            popup='Start',
            icon=folium.Icon(color='blue', icon='play', prefix='fa')
        ).add_to(m)
        
        # End marker
        folium.Marker(
            location=[gps_df.iloc[-1]['latitude'], gps_df.iloc[-1]['longitude']],
            popup='End',
            icon=folium.Icon(color='purple', icon='stop', prefix='fa')
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add minimap
    plugins.MiniMap().add_to(m)
    
    # Add fullscreen option
    plugins.Fullscreen().add_to(m)
    
    # Save map
    m.save(output_html_path)
    print(f"\nInteractive map saved to: {output_html_path}")
    print("="*70)
    
    # Open map in browser
    if show:
        try:
            print(f"Opening map in browser...")
            webbrowser.open('file://' + os.path.abspath(output_html_path))
            print("Map opened successfully!")
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            print(f"Please open manually: {output_html_path}")

