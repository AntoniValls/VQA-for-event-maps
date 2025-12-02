"""
Script for creating the ground-truth labels of the sequences
"""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import os
import json
import cv2
from PIL import Image
from datetime import datetime

from core.promptManager import PromptManager

def display_image_with_text(image, text, window_name="Ground Truth Labeling"):
    """
    Display image with text overlay for labeling.
    
    Args:
        image: PIL Image
        text: Text to display
        window_name: OpenCV window name
    """
    # Convert PIL to OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Create a larger canvas for text
    h, w = img_cv.shape[:2]
    text_height = 150
    canvas = np.zeros((h + text_height, w, 3), dtype=np.uint8)
    canvas[text_height:, :] = img_cv
    
    # Add text
    y_offset = 30
    for i, line in enumerate(text.split('\n')):
        cv2.putText(canvas, line, (10, y_offset + i*35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow(window_name, canvas)


def load_existing_labels(output_path):
    """
    Load existing labels from file.
    
    Returns:
        set: Set of image names that have been labeled
    """
    labeled_images = set()
    
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        label = json.loads(line.strip())
                        labeled_images.add(label.get('image_name'))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Error reading existing labels: {e}")
    
    return labeled_images


def create_ground_truth_labels(input_dir, prompt_preset="full_hierarchical", num_keyframes=20, override_existing=False):
    """
    Manual ground truth labeling tool.
    
    Args:
        input_dir: Directory containing images folder
        prompt_preset: Question preset to use
        num_keyframes: Number of images to label
        override_existing: If True, re-label already labeled images. If False, skip them.
    """
    import numpy as np
    
    # Setup paths
    image_dir = os.path.join(input_dir, "images")
    output_path = os.path.join(input_dir, "ground_truth_labels.jsonl")
    
    print(f"="*70)
    print(f"GROUND TRUTH LABELING TOOL")
    print(f"="*70)
    print(f"Input: {image_dir}")
    print(f"Output: {output_path}")
    print(f"Preset: {prompt_preset}")
    print(f"Override existing: {override_existing}")
    print(f"="*70)
    
    # Load existing labels
    already_labeled = set()
    if not override_existing and os.path.exists(output_path):
        already_labeled = load_existing_labels(output_path)
        if already_labeled:
            print(f"\nFound {len(already_labeled)} already labeled images (will skip)")
        else:
            print(f"\nNo existing labels found (starting fresh)")
    elif override_existing and os.path.exists(output_path):
        print(f"\nWARNING: Override mode enabled - will re-label existing images")
        response = input("Continue? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Aborted.")
            return 1
    else:
        print(f"\nNo existing labels file found (starting fresh)")
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))
        image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return 1
    
    print(f"Found {len(image_files)} images")
    
    # Calculate stride
    frame_stride = max(1, int(len(image_files) / num_keyframes))
    print(f"Will label every {frame_stride} image(s) → ~{len(range(0, len(image_files), frame_stride))} images\n")
    
    # Load prompts
    prompt_json_path = "../inout/vqa_prompts.json"
    prompt_manager = PromptManager(str(prompt_json_path), preset=prompt_preset)
    prompt_manager.print_hierarchy_info()
    
    # Instructions
    print("\n" + "="*70)
    print("INSTRUCTIONS")
    print("="*70)
    print("For each question:")
    print("  - Press 'y' or 'Y' for YES")
    print("  - Press 'n' or 'N' for NO")
    print("  - Press 's' or 'S' to SKIP current image")
    print("  - Press 'q' or 'Q' to QUIT and save progress")
    print("="*70)
    input("\nPress ENTER to start labeling...")
    
    # Create window
    cv2.namedWindow("Ground Truth Labeling", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ground Truth Labeling", 1280, 850)
    
    # Track progress
    labeled_count = 0
    skipped_count = 0
    already_labeled_count = 0
    
    try:
        # Determine write mode
        write_mode = "w" if override_existing else "a"
        
        with open(output_path, write_mode, encoding="utf-8") as f:
            for idx, image_path in enumerate(image_files):
                if idx % frame_stride != 0:
                    continue
                
                # Check if already labeled
                if not override_existing and image_path.name in already_labeled:
                    already_labeled_count += 1
                    print(f"\n[SKIP] Image {idx+1}/{len(image_files)}: {image_path.name} (already labeled)")
                    continue
                
                # Load image
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
                    continue
                
                # Reset for new image
                prompt_manager.reset_answer_history()
                image_labels = []
                skip_image = False
                
                print(f"\n{'='*70}")
                print(f"IMAGE {idx+1}/{len(image_files)}: {image_path.name}")
                print(f"Progress: {labeled_count} labeled, {skipped_count} skipped, {already_labeled_count} already done")
                print(f"{'='*70}\n")
                
                initial_prompts = prompt_manager.get_initial_prompts()
                
                for q_num, question_data in enumerate(initial_prompts, 1):
                    if skip_image:
                        break
                    
                    question_id = question_data['id']
                    question_text = question_data['text']
                    short_label = prompt_manager.get_short_label(question_data)
                    
                    # Display image with question
                    display_text = f"Question {q_num}/{len(initial_prompts)}: {short_label}\n{question_text}\n\nPress: Y=Yes | N=No | S=Skip | Q=Quit"
                    display_image_with_text(image, display_text, "Ground Truth Labeling")
                    
                    # Get answer
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        
                        if key == ord('y') or key == ord('Y'):
                            answer = "yes"
                            print(f"  [{q_num}/{len(initial_prompts)}] {short_label}: YES")
                            break
                        elif key == ord('n') or key == ord('N'):
                            answer = "no"
                            print(f"  [{q_num}/{len(initial_prompts)}] {short_label}: NO")
                            break
                        elif key == ord('s') or key == ord('S'):
                            print(f"\n  >>> SKIPPING IMAGE <<<\n")
                            skip_image = True
                            break
                        elif key == ord('q') or key == ord('Q'):
                            print(f"\n  >>> QUITTING - Progress saved <<<\n")
                            cv2.destroyAllWindows()
                            return 0
                    
                    if skip_image:
                        break
                    
                    # Store label
                    label_obj = {
                        "image_path": str(image_path),
                        "image_name": image_path.name,
                        "image_index": idx,
                        "question_id": question_id,
                        "question": question_text,
                        "answer": answer,
                        "level": 1,
                        "short_label": short_label,
                        "timestamp": datetime.now().isoformat()
                    }
                    image_labels.append(label_obj)
                    
                    # Get follow-ups if yes
                    if answer == "yes":
                        followups = prompt_manager.get_followup_prompts(question_id, answer)
                        
                        if followups:
                            print(f"\n  Follow-up questions for '{short_label}':")
                            
                            for fq_num, followup_q in enumerate(followups, 1):
                                followup_id = followup_q['id']
                                followup_text = followup_q['text']
                                followup_label = prompt_manager.get_short_label(followup_q)
                                
                                # Display image with follow-up question
                                display_text = f"Follow-up {fq_num}/{len(followups)} for '{short_label}':\n{followup_label}\n{followup_text}\n\nPress: Y=Yes | N=No | S=Skip | Q=Quit"
                                display_image_with_text(image, display_text, "Ground Truth Labeling")
                                
                                # Get answer
                                while True:
                                    key = cv2.waitKey(0) & 0xFF
                                    
                                    if key == ord('y') or key == ord('Y'):
                                        followup_answer = "yes"
                                        print(f"    └─ [{fq_num}/{len(followups)}] {followup_label}: YES")
                                        break
                                    elif key == ord('n') or key == ord('N'):
                                        followup_answer = "no"
                                        print(f"    └─ [{fq_num}/{len(followups)}] {followup_label}: NO")
                                        break
                                    elif key == ord('s') or key == ord('S'):
                                        print(f"\n  >>> SKIPPING IMAGE <<<\n")
                                        skip_image = True
                                        break
                                    elif key == ord('q') or key == ord('Q'):
                                        print(f"\n  >>> QUITTING - Progress saved <<<\n")
                                        cv2.destroyAllWindows()
                                        return 0
                                
                                if skip_image:
                                    break
                                
                                # Store follow-up label
                                followup_label_obj = {
                                    "image_path": str(image_path),
                                    "image_name": image_path.name,
                                    "image_index": idx,
                                    "question_id": followup_id,
                                    "question": followup_text,
                                    "answer": followup_answer,
                                    "level": 2,
                                    "parent_question": question_id,
                                    "short_label": followup_label,
                                    "timestamp": datetime.now().isoformat()
                                }
                                image_labels.append(followup_label_obj)

                                # Get follow-ups level 3 if yes
                                if followup_answer == "yes":
                                    ffollowups = prompt_manager.get_followup_prompts(followup_id, followup_answer)
                                    
                                    if ffollowups:
                                        print(f"\n  Follow-up questions for '{followup_label}':")
                                        
                                        for ffq_num, ffollowup_q in enumerate(ffollowups, 1):
                                            ffollowup_id = ffollowup_q['id']
                                            ffollowup_text = ffollowup_q['text']
                                            ffollowup_label = prompt_manager.get_short_label(ffollowup_q)
                                            
                                            # Display image with ffollow-up question
                                            display_text = f"Follow-up {ffq_num}/{len(ffollowups)} for '{ffollowup_label}':\n{ffollowup_label}\n{ffollowup_text}\n\nPress: Y=Yes | N=No | S=Skip | Q=Quit"
                                            display_image_with_text(image, display_text, "Ground Truth Labeling")
                                            
                                            # Get answer
                                            while True:
                                                key = cv2.waitKey(0) & 0xFF
                                                
                                                if key == ord('y') or key == ord('Y'):
                                                    ffollowup_answer = "yes"
                                                    print(f"    └─ [{ffq_num}/{len(ffollowups)}] {ffollowup_label}: YES")
                                                    break
                                                elif key == ord('n') or key == ord('N'):
                                                    ffollowup_answer = "no"
                                                    print(f"    └─ [{ffq_num}/{len(ffollowups)}] {ffollowup_label}: NO")
                                                    break
                                                elif key == ord('s') or key == ord('S'):
                                                    print(f"\n  >>> SKIPPING IMAGE <<<\n")
                                                    skip_image = True
                                                    break
                                                elif key == ord('q') or key == ord('Q'):
                                                    print(f"\n  >>> QUITTING - Progress saved <<<\n")
                                                    cv2.destroyAllWindows()
                                                    return 0
                                            
                                            if skip_image:
                                                break
                                            
                                            # Store follow-up label
                                            ffollowup_label_obj = {
                                                "image_path": str(image_path),
                                                "image_name": image_path.name,
                                                "image_index": idx,
                                                "question_id": ffollowup_id,
                                                "question": ffollowup_text,
                                                "answer": ffollowup_answer,
                                                "level": 3,
                                                "parent_question": followup_id,
                                                "short_label": ffollowup_label,
                                                "timestamp": datetime.now().isoformat()
                                            }
                                            image_labels.append(ffollowup_label_obj)
                                        
                                        if skip_image:
                                            break
                            if skip_image:
                                break
                
                # Save labels for this image
                if not skip_image:
                    for label in image_labels:
                        f.write(json.dumps(label) + "\n")
                    f.flush()
                    labeled_count += 1
                    print(f"\n✓ Saved {len(image_labels)} labels for {image_path.name}")
                else:
                    skipped_count += 1
    
    finally:
        cv2.destroyAllWindows()
    
    print(f"\n{'='*70}")
    print(f"LABELING COMPLETE")
    print(f"{'='*70}")
    print(f"Total images labeled: {labeled_count}")
    print(f"Total images skipped: {skipped_count}")
    print(f"Already labeled (skipped): {already_labeled_count}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    # ============ CONFIGURATION ============
    PROMPT_PRESET = "full_hierarchical"  # Options: level_1_only, full_hierarchical, crossing, etc.
    CONTINENT = "America"
    CITY = "BuenosAires"
    NUM_KEYFRAMES = 20  # Number of images to label
    OVERRIDE_EXISTING = False  # Set to True to re-label already labeled images
    # =======================================
    
    print(f"="*70)
    print(f"Ground Truth Labeling Tool - Hierarchical VQA")
    print(f"="*70)
    print(f"Preset: {PROMPT_PRESET}")
    print(f"City: {CITY}, {CONTINENT}")
    print(f"Target labels: ~{NUM_KEYFRAMES} images")
    print(f"="*70)
    
    input_directory = f"../data/{CONTINENT}/{CITY}"
    
    exit_code = create_ground_truth_labels(input_directory, PROMPT_PRESET, NUM_KEYFRAMES, OVERRIDE_EXISTING)
    sys.exit(exit_code)