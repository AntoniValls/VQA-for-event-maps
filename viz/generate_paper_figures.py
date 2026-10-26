import os
import json
import base64
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import string


def get_normalized_risk(img_questions, primary_ids):
    """
    Calculates a normalized risk score [0, 1] using a weighted penalty/reward system.
    Copied from original code for consistency.
    """
    HAZARD_CONFIG = {
        "CRITICAL": {"weight": 1, "ids": ["q_construction_visible", "q_surface_hazardous", "q_pedestrian_not_on_sidewalk"]},
        "HIGH":     {"weight": 0.6, "ids": ["q_crossing_nearby", "q_stairs_visible", "q_obstacle_blocking"]},
        "LOW":      {"weight": 0.3, "ids": ["q_pedestrians_present", "q_vehicle_nearby"]}
    }
    
    SAFETY_REWARD_RATIO = 1/8
    weight_lookup = {qid: cfg["weight"] for cfg in HAZARD_CONFIG.values() for qid in cfg["ids"]}
    active_weights = [weight_lookup.get(qid, 0) for qid in primary_ids]
    max_theoretical = sum(active_weights)
    
    if max_theoretical <= 0:
        return 0.0

    net_score = 0.0
    for q_id in primary_ids:
        if q_id not in img_questions:
            continue
            
        ans = img_questions[q_id].get('answer', '').lower().strip().translate(str.maketrans('', '', string.punctuation))
        weight = weight_lookup.get(q_id, 0)
        
        is_yes = any(pos in ans for pos in ['yes', 'true', 'hazard'])
        is_no = any(neg in ans for neg in ['no', 'false', 'safe'])
        
        if is_yes:
            net_score += weight
        elif is_no:
            net_score -= (weight * SAFETY_REWARD_RATIO)

    normalized = max(0, net_score) / max_theoretical
    return min(normalized, 1.0)


def get_color_from_normalized(score):
    """Return matplotlib-compatible color based on risk score."""
    if score <= 0.15: return '#2ecc71'     # Green (Very Safe)
    if score < 0.4: return '#f1c40f'       # Yellow
    if score < 0.7: return '#e67e22'       # Orange
    return '#e74c3c'                       # Red


def generate_paper_figure_panel(
    image_path,
    answers_dict,
    question_full_text,
    question_to_category,
    category_names,
    primary_ids,
    output_path,
    figsize=(12, 8),
    dpi=300,
    show_risk_score=True,
    font_size=9,
    title_font_size=11
):
    """
    Generate a single publication-quality figure panel with image and analysis table.
    
    Parameters:
    -----------
    image_path : str or Path
        Path to the image file
    answers_dict : dict
        Dictionary of question_id -> answer data
    question_full_text : dict
        Mapping of question_id to full question text
    question_to_category : dict
        Mapping of question_id to category_id
    category_names : dict
        Mapping of category_id to category display name
    primary_ids : list
        List of primary question IDs for risk calculation
    output_path : str or Path
        Where to save the figure
    figsize : tuple
        Figure size in inches (width, height)
    dpi : int
        Resolution for output image
    show_risk_score : bool
        Whether to show the computed risk score
    font_size : int
        Base font size for text
    title_font_size : int
        Font size for titles
    """
    
    # Create figure with custom layout
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 0.8], width_ratios=[1.2, 1],
                  hspace=0.3, wspace=0.3, left=0.08, right=0.95, top=0.93, bottom=0.05)
    
    # Load and display image
    img = Image.open(image_path)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(img)
    ax_img.axis('off')
    
    # Calculate risk score
    risk_score = get_normalized_risk(answers_dict, primary_ids)
    risk_color = get_color_from_normalized(risk_score)
    
    # Add image title with risk indicator
    img_name = Path(image_path).name
    if show_risk_score:
        title_text = f"{img_name}\nRisk Score: {risk_score:.3f}"
        ax_img.set_title(title_text, fontsize=title_font_size, fontweight='bold', 
                        pad=10, color=risk_color)
    else:
        ax_img.set_title(img_name, fontsize=title_font_size, fontweight='bold', pad=10)
    
    # Organize answers by category
    cat_data = {}
    for q_id, ans_data in answers_dict.items():
        cid = question_to_category.get(q_id, 'Other')
        if cid not in cat_data:
            cat_data[cid] = []
        
        question_text = question_full_text.get(q_id, q_id)
        answer = ans_data.get('answer', 'N/A')
        confidence = ans_data.get('confidence')
        
        cat_data[cid].append({
            'question': question_text,
            'answer': answer,
            'confidence': confidence,
            'q_id': q_id
        })
    
    # Sort categories: level_1 first, then others alphabetically
    sorted_categories = sorted(cat_data.keys(), 
                              key=lambda x: (0 if 'level_1' in x.lower() else 1, x))
    
    # Create tables for each category
    num_categories = len(sorted_categories)
    for idx, cid in enumerate(sorted_categories):
        if idx < 2:
            ax_table = fig.add_subplot(gs[idx, 1])
        else:
            # If more than 2 categories, you might need to adjust layout
            break
        
        # Prepare table data
        table_data = []
        cell_colors = []
        
        for item in cat_data[cid]:
            answer = item['answer']
            
            # Determine cell color based on answer
            if any(pos in answer.lower() for pos in ['yes', 'true', 'hazard', 'danger']):
                answer_color = '#ffcccc'  # Light red
            elif any(neg in answer.lower() for neg in ['no', 'false', 'safe']):
                answer_color = '#ccffcc'  # Light green
            else:
                answer_color = '#ffffff'  # White
            
            # Format answer with confidence if available
            if item['confidence'] is not None:
                answer_text = f"{answer}\n({item['confidence']:.2f})"
            else:
                answer_text = answer
            
            table_data.append([item['question'], answer_text])
            cell_colors.append([['#f0f0f0'], [answer_color]])
        
        # Create table
        table = ax_table.table(
            cellText=table_data,
            colLabels=['Question', 'Answer'],
            cellLoc='left',
            loc='center',
            colWidths=[0.6, 0.4]
        )
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1, 2)
        
        # Apply colors and styling
        for i in range(len(table_data)):
            for j in range(2):
                cell = table[(i+1, j)]
                cell.set_facecolor(cell_colors[i][j][0])
                cell.set_edgecolor('#cccccc')
                cell.set_linewidth(0.5)
                
                # Make question text smaller and wrap
                if j == 0:
                    cell.set_text_props(fontsize=font_size-1, wrap=True)
                else:
                    cell.set_text_props(fontsize=font_size, weight='bold', wrap=True)
        
        # Style header
        for j in range(2):
            cell = table[(0, j)]
            cell.set_facecolor('#4a5568')
            cell.set_text_props(color='white', weight='bold', fontsize=font_size+1)
            cell.set_edgecolor('#2d3748')
            cell.set_linewidth(1)
        
        ax_table.axis('off')
        ax_table.set_title(category_names.get(cid, cid), 
                          fontsize=title_font_size, fontweight='bold', pad=10)
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved figure to {output_path}")


def generate_multi_image_comparison(
    image_configs,
    question_full_text,
    question_to_category,
    category_names,
    primary_ids,
    output_path,
    ncols=2,
    figsize=(20, 14),
    dpi=300,
    show_category='level_1',  # Which category to display in comparison
    font_size=7
):
    """
    Generate a multi-panel comparison figure with multiple images from different locations.
    
    Parameters:
    -----------
    image_configs : list of dict
        List of image configurations, each containing:
        - 'image_path': path to the image
        - 'answers': dict of answers for this image
        - 'continent': continent name
        - 'city': city name
    question_full_text : dict
        Mapping of question_id to full question text
    question_to_category : dict
        Mapping of question_id to category_id
    category_names : dict
        Mapping of category_id to category display name
    primary_ids : list
        List of primary question IDs for risk calculation
    output_path : str or Path
        Where to save the figure
    ncols : int
        Number of columns in the grid
    figsize : tuple
        Figure size in inches (width, height)
    dpi : int
        Resolution for output image
    show_category : str
        Category ID to focus on for the comparison table (e.g., 'level_1' for primary hazards)
    font_size : int
        Base font size for annotations
    """
    
    num_images = len(image_configs)
    nrows = int(np.ceil(num_images / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    if num_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, config in enumerate(image_configs):
        ax = axes[idx]
        img_path = config['image_path']
        img_answers = config['answers']
        continent = config.get('continent', 'Unknown')
        city = config.get('city', 'Unknown')
        
        # Load and display image
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
        
        # Calculate risk score
        risk_score = get_normalized_risk(img_answers, primary_ids)
        risk_color = get_color_from_normalized(risk_score)
        
        # Create annotation text for key questions - SHOW COMPLETE QUESTIONS
        annotation_lines = [f"Risk Score: {risk_score:.3f}"]
        annotation_lines.append("-" * 50)
        
        # Get questions from specified category
        for q_id, ans_data in img_answers.items():
            cid = question_to_category.get(q_id, '')
            if show_category and show_category.lower() not in cid.lower():
                continue
            
            answer = ans_data.get('answer', 'N/A')
            question_full = question_full_text.get(q_id, q_id)
            
            # Show COMPLETE question with answer
            annotation_lines.append(f"Q: {question_full}")
            annotation_lines.append(f"A: {answer}")
            annotation_lines.append("")  # Blank line for spacing
        
        # Add text box with answers
        textstr = '\n'.join(annotation_lines)
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, 
                    edgecolor=risk_color, linewidth=3, pad=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=font_size,
                verticalalignment='top', bbox=props, family='monospace',
                linespacing=1.3)
        
        # Add title with Continent - City format
        title_text = f"{continent} - {city}"
        ax.set_title(title_text, fontsize=12, fontweight='bold', 
                    color=risk_color, pad=8)
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved comparison figure to {output_path}")


def create_hierarchical_question_display(
    continent,
    city,
    model,
    image_index,
    output_filename="hierarchical_display.png",
    prompt_json_path="../inout/vqa_prompts.json",
    dpi=300,
    figsize=(18, 12)
):
    """
    Create a figure showing the hierarchical nature of the VQA dataset.
    Displays one image with Level 1 questions prominently, and crossing questions
    as follow-ups with visual hierarchy.
    
    Parameters:
    -----------
    continent : str
        Continent name (e.g., 'Asia')
    city : str
        City name (e.g., 'Bombai')
    model : str
        Model name (e.g., 'qwen-vl')
    image_index : int
        Which image to use from the location
    output_filename : str
        Name of output file (saved in ../data/model_comparison/)
    """
    
    # Load configuration
    primary_ids = []
    question_full_text, question_to_category, category_names = {}, {}, {}
    category_descriptions = {}
    
    if os.path.exists(prompt_json_path):
        with open(prompt_json_path, 'r') as f:
            prompt_config = json.load(f)
            primary_ids = prompt_config.get('prompt_presets', {}).get('level_1_only', {}).get('enabled_questions', [])
            
            for cat_id, cat_data in prompt_config['prompt_categories'].items():
                category_names[cat_id] = cat_data.get('name', cat_id)
                category_descriptions[cat_id] = cat_data.get('description', '')
                for q in cat_data['questions']:
                    question_full_text[q['id']] = q['text']
                    question_to_category[q['id']] = cat_id
    
    # Load data
    gps_csv_path = f"../data/{continent}/{city}/gps_positions.csv"
    answers_path = os.path.join(Path(gps_csv_path).parent, f"results/{model}/answers.jsonl")
    image_dir = os.path.join(Path(gps_csv_path).parent, "images_selected")
    
    gps_df = pd.read_csv(gps_csv_path).sort_values(by='filename')
    
    with open(answers_path, 'r') as f:
        answers_data = [json.loads(line) for line in f]
    
    answers_by_image = {}
    for ans in answers_data:
        img_name = ans['image_name']
        if img_name not in answers_by_image:
            answers_by_image[img_name] = {}
        answers_by_image[img_name][ans['question_id']] = ans
    
    # Get specific image
    image_list = [img for img in gps_df['filename'].tolist() if img in answers_by_image]
    if image_index >= len(image_list):
        image_index = len(image_list) - 1
    
    img_name = image_list[image_index]
    img_path = Path(image_dir) / img_name
    img_answers = answers_by_image[img_name]
    
    # Calculate risk score
    risk_score = get_normalized_risk(img_answers, primary_ids)
    risk_color = get_color_from_normalized(risk_score)
    
    
    # ---- Build preset structure from prompt config ----
    preset_map = {}
    trigger_map = {}  # which Level 1 question activates which preset

    for preset_id, preset_data in prompt_config["prompt_presets"].items():
        if preset_id == "level_1_only":
            continue
        if preset_data["enabled_questions"] == "all":
            continue
        
        enabled = preset_data["enabled_questions"]
        preset_map[preset_id] = enabled
        
        # first question is always the Level 1 trigger
        trigger_map[preset_id] = enabled[0]
    
    # ---- Split Level 1 and Conditional Questions ----
    level_1_questions = {}
    conditional_groups = {k: {} for k in preset_map.keys()}

    for q_id, ans_data in img_answers.items():
        if q_id in primary_ids:
            level_1_questions[q_id] = ans_data
        
        for preset_id, enabled_list in preset_map.items():
            if q_id in enabled_list and q_id != trigger_map[preset_id]:
                conditional_groups[preset_id][q_id] = ans_data


    # ---- Paper-Optimized Compact Layout ----
    fig = plt.figure(figsize=(12, 6.5), dpi=dpi)
    gs = GridSpec(
        1, 2,
        figure=fig,
        width_ratios=[1.25, 1],
        wspace=0.01,
        left=0.01,
        right=0.99,
        top=0.92,
        bottom=0.06
    )

    # LEFT: Image
    img = Image.open(img_path)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.axis('off')

    title_text = f"Qwen-VL – Tokio | Predicted Risk Score {risk_score:.3f}"
    ax_img.set_title(
        title_text,
        fontsize=12,
        fontweight='bold',
        color=risk_color,
        pad=6
    )

    # RIGHT: Unified Hierarchical Text Block
    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.axis('off')

    text_block = ""

    # ------------------
    # LEVEL 1
    # ------------------
    text_block += "Level 1\n"
    text_block += "─" * 40 + "\n"

    for q_id, ans_data in level_1_questions.items():
        question = question_full_text.get(q_id, q_id)
        answer = ans_data.get('answer', 'N/A')

        if "yes" in answer.lower():
            symbol = "!"
        elif "no" in answer.lower():
            symbol = "✓"
        else:
            symbol = "•"

        text_block += f"{symbol} {question}\n"
        text_block += f"    {answer}\n"

    # ------------------
    # CONDITIONAL
    # ------------------
    text_block += "\nLevel 2 + 3\n"
    text_block += "─" * 40 + "\n"

    triggered_any = False

    for preset_id, questions in conditional_groups.items():

        trigger_q = trigger_map[preset_id]
        trigger_answer = img_answers.get(trigger_q, {}).get("answer", "")

        if "yes" not in trigger_answer.lower():
            continue

        triggered_any = True
        preset_title = preset_id.replace("_", " ").title()

        text_block += f"\n[{preset_title}]\n"

        for q_id, ans_data in questions.items():
            question = question_full_text.get(q_id, q_id)
            answer = ans_data.get('answer', 'N/A')

            if "yes" in answer.lower():
                symbol = "!"
            elif "no" in answer.lower():
                symbol = "✓"
            else:
                symbol = "•"

            text_block += f"  {symbol} {question}\n"
            text_block += f"      {answer}\n"

    if not triggered_any:
        text_block += "\n(None triggered)"

    # Draw compact box
    ax_text.text(
        0.01, 0.99,
        text_block,
        transform=ax_text.transAxes,
        fontsize=7.2,
        verticalalignment='top',
        family='monospace',
        linespacing=1.15,
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            edgecolor=risk_color,
            linewidth=1.2,
            pad=0.4
        )
    )

    # Save output
    output_dir = Path("../data/model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Hierarchical display saved to: {output_path}")
    return output_path


def create_location_comparison_figure(
    location_configs,
    output_filename="location_comparison.png",
    prompt_json_path="../inout/vqa_prompts.json",
    ncols=2,
    dpi=300,
    figsize=(20, 14),
    show_category='level_1'
):
    """
    Create a comparison figure from images across different locations.
    
    Parameters:
    -----------
    location_configs : list of dict
        Each dict should contain:
        - 'continent': str (e.g., 'Asia')
        - 'city': str (e.g., 'Bombai')
        - 'model': str (e.g., 'qwen-vl')
        - 'image_index': int (which image from that location to use)
    output_filename : str
        Name of the output file (saved in ../data/model_comparison/)
    prompt_json_path : str
        Path to the VQA prompts configuration file
    ncols : int
        Number of columns in the comparison grid
    dpi : int
        Resolution for output
    figsize : tuple
        Figure size (width, height)
    show_category : str
        Which category of questions to display (e.g., 'level_1')
    
    Example:
    --------
    create_location_comparison_figure([
        {'continent': 'Asia', 'city': 'Bombai', 'model': 'qwen-vl', 'image_index': 10},
        {'continent': 'Europe', 'city': 'Paris', 'model': 'qwen-vl', 'image_index': 5},
        {'continent': 'North America', 'city': 'New York', 'model': 'qwen-vl', 'image_index': 15},
        {'continent': 'Africa', 'city': 'Cairo', 'model': 'qwen-vl', 'image_index': 8}
    ])
    """
    
    # Load configuration
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
    
    # Prepare image configs
    image_configs = []
    
    for loc_config in location_configs:
        continent = loc_config['continent']
        city = loc_config['city']
        model = loc_config['model']
        img_idx = loc_config['image_index']
        
        # Build paths
        gps_csv_path = f"../data/{continent}/{city}/gps_positions.csv"
        answers_path = os.path.join(Path(gps_csv_path).parent, f"results/{model}/answers.jsonl")
        image_dir = os.path.join(Path(gps_csv_path).parent, "images_selected")
        
        # Load GPS data to get image filename
        gps_df = pd.read_csv(gps_csv_path).sort_values(by='filename')
        
        # Load answers
        with open(answers_path, 'r') as f:
            answers_data = [json.loads(line) for line in f]
        
        answers_by_image = {}
        for ans in answers_data:
            img_name = ans['image_name']
            if img_name not in answers_by_image:
                answers_by_image[img_name] = {}
            answers_by_image[img_name][ans['question_id']] = ans
        
        # Get the specific image
        image_list = [img for img in gps_df['filename'].tolist() if img in answers_by_image]
        
        if img_idx >= len(image_list):
            print(f"Warning: Index {img_idx} out of range for {continent}/{city}. Using last image.")
            img_idx = len(image_list) - 1
        
        img_name = image_list[img_idx]
        img_path = Path(image_dir) / img_name
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue
        
        image_configs.append({
            'image_path': img_path,
            'answers': answers_by_image[img_name],
            'continent': continent,
            'city': city
        })
    
    # Create output directory
    output_dir = Path("../data/model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    
    # Generate the comparison figure
    generate_multi_image_comparison(
        image_configs=image_configs,
        question_full_text=question_full_text,
        question_to_category=question_to_category,
        category_names=category_names,
        primary_ids=primary_ids,
        output_path=output_path,
        ncols=ncols,
        figsize=figsize,
        dpi=dpi,
        show_category=show_category
    )
    
    print(f"\n✓ Comparison figure saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    
    # Example 1: Show hierarchical question structure with one image
    print("=" * 70)
    print("Generating hierarchical question display...")
    print("=" * 70)
    
    create_hierarchical_question_display(
        continent='Asia',
        city='Tokio2',
        model='qwen-vl',
        image_index=1,
        output_filename="hierarchical_structure.png"
    )