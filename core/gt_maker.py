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
import shutil

from core.promptManager import PromptManager

import shutil

def select_keyframes_interactively_strided(
    image_files,
    target_count=20,
    output_dir=None,
    select_images=True,
    window_name="Keyframe Selection"
):
    """
    Select keyframes with one image per stride segment.

    If select_images=True:
        - Interactive selection
        - Copies selected images to output_dir

    If select_images=False:
        - Loads images directly from output_dir
        - No UI shown

    Returns:
        List[Path]: selected image paths (original paths if selecting,
                    paths in output_dir if loading)
    """
    assert output_dir is not None, "output_dir must be provided"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # MODE 1: load already selected images
    # ---------------------------------------------------------
    if not select_images:
        selected = sorted(
            p for p in output_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        )

        if not selected:
            raise RuntimeError(
                f"select_images=False but no images found in {output_dir}"
            )

        print(f"Loaded {len(selected)} pre-selected images from {output_dir}")
        return selected[:target_count]

    # ---------------------------------------------------------
    # MODE 2: interactive strided selection
    # ---------------------------------------------------------
    assert target_count > 0

    total_images = len(image_files)
    stride = total_images // target_count

    selected = []

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 850)

    for seg_idx in range(target_count):
        start = seg_idx * stride
        end = total_images if seg_idx == target_count - 1 else (seg_idx + 1) * stride
        segment_images = image_files[start:end]

        if not segment_images:
            continue

        print(f"\nSegment {seg_idx + 1}/{target_count} "
              f"(frames {start}–{end - 1})")

        kept_in_segment = False

        for image_path in segment_images:
            image = Image.open(image_path).convert("RGB")

            text = (
                f"Segment {seg_idx + 1}/{target_count}\n"
                f"Frames {start}–{end - 1}\n"
                f"Selected: {len(selected)}/{target_count}\n\n"
                "K = keep (advance segment)\n"
                "S = skip (next image)\n"
                "Q = quit"
            )

            display_image_with_text(image, text, window_name)

            while True:
                key = cv2.waitKey(0) & 0xFF

                if key in (ord("k"), ord("K")):
                    selected.append(image_path)

                    dst = output_dir / image_path.name
                    if not dst.exists():
                        shutil.copy2(image_path, dst)

                    print(f"[KEEP] {image_path.name}")
                    kept_in_segment = True
                    break

                elif key in (ord("s"), ord("S")):
                    print(f"[SKIP] {image_path.name}")
                    break

                elif key in (ord("q"), ord("Q")):
                    print("Selection aborted early.")
                    cv2.destroyAllWindows()
                    return selected

            if kept_in_segment:
                break

        if not kept_in_segment:
            print("⚠ No image selected in this segment.")

    cv2.destroyAllWindows()

    print(f"\nFinal selection: {len(selected)} images")
    print(f"Copied to: {output_dir}")
    return selected

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
    
    return 
    
def load_existing_labels_by_image_and_question(output_path):
    """
    Returns:
        dict[str, set[str]]: image_name -> set(question_id)
    """
    labels = {}

    if not os.path.exists(output_path):
        return labels

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                img = obj.get("image_name")
                qid = obj.get("question_id")
                if img and qid:
                    labels.setdefault(img, set()).add(qid)
            except json.JSONDecodeError:
                continue

    return labels


def create_ground_truth_labels(
    input_dir, 
    prompt_preset="full_hierarchical", 
    num_keyframes=20, 
    override_existing=False, 
    select_images=False, 
    patch_questions=None):

    """
    Manual ground truth labeling tool.
    
    Args:
        input_dir: Directory containing images folder
        prompt_preset: Question preset to use
        num_keyframes: Number of images to label
        override_existing: If True, re-label already labeled images. If False, skip them.
        patch_questions:
        - None → normal behavior
        - set(question_id) → only ask missing questions in this set
    """
    
    # ------------------------------------------------------------------
    # Setup paths
    # ------------------------------------------------------------------

    image_dir = os.path.join(input_dir, "images")
    output_path = os.path.join(input_dir, "ground_truth_labels.jsonl")
    
    print(f"="*70)
    print(f"GROUND TRUTH LABELING TOOL")
    print(f"="*70)
    print(f"Input: {image_dir}")
    print(f"Output: {output_path}")
    print(f"Preset: {prompt_preset}")
    print(f"Override existing: {override_existing}")
    print(f"Patch mode: {patch_questions is not None}")
    if patch_questions:
        print(f"Patching questions: {patch_questions}")
    print(f"="*70)
    
    # ------------------------------------------------------------------
    # Load existing labels
    # ------------------------------------------------------------------
    existing_labels = load_existing_labels_by_image_and_question(output_path)
    if os.path.exists(output_path) and not override_existing:
        existing_labels = load_existing_labels_by_image_and_question(output_path)

    # ------------------------------------------------------------------
    # Prepare output file
    # ------------------------------------------------------------------
    if override_existing:
        print("⚠ Override mode enabled: existing labels will be replaced")
        response = input("Continue? (y/n): ").strip().lower()
        if response not in {"y", "yes"}:
            print("Aborted.")
            return 1
        write_mode = "w"
    else:
        write_mode = "a"
    
    # ------------------------------------------------------------------
    # Collect images
    # ------------------------------------------------------------------
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))
        image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))

    image_files = sorted(image_files)
    if not image_files:
        print("No images found.")
        return 1
    
    print(f"Found {len(image_files)} images")
    
    # ------------------------------------------------------------------
    # Select keyframes
    # ------------------------------------------------------------------
    selected_dir = os.path.join(input_dir, "images_selected")

    # If no images in the folder trigger select_images=True
    if not os.path.isdir(selected_dir) or sum(1 for p in Path(selected_dir).iterdir() if p.is_file()) != num_keyframes:
        select_images = True
        print("Not enough selected images in the cache. Triggering selection")

    print("\nSelecting strided keyframes interactively...")
    selected_images = select_keyframes_interactively_strided(
        image_files=image_files,
        target_count=num_keyframes,
        output_dir=selected_dir,
        select_images=select_images
    )

    if not selected_images:
        print("No images selected. Exiting.")
        return 1

    # ------------------------------------------------------------------
    # Load prompts
    # ------------------------------------------------------------------
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
        with open(output_path, write_mode, encoding="utf-8") as f:
            for idx, image_path in enumerate(selected_images):
                original_index = image_files.index(image_path.parent.parent / "images" / image_path.name)

                # Get the image already answered questions
                answered = existing_labels.get(image_path.name, set())
                
                # Load image
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
                    continue
                
                # Reset for new image
                prompt_manager.reset_answer_history()
                initial_prompts = prompt_manager.get_initial_prompts()
                image_labels = []
                skip_image = False

                # ----------------------------------------------------------
                # Determine which questions to ask
                # ----------------------------------------------------------
                if patch_questions is not None:
                    # Patch or redo only selected questions
                    initial_prompts = [
                        q for q in initial_prompts
                        if q["id"] in patch_questions
                        and (override_existing or q["id"] not in answered)
                    ]
                elif not override_existing:
                    # Normal incremental mode
                    initial_prompts = [
                        q for q in initial_prompts
                        if q["id"] not in answered
                    ]

                if not initial_prompts:
                    continue

                print(f"\n{'='*70}")
                print(f"IMAGE {original_index+1}/{len(image_files)}: {image_path.name}")
                print(f"{'='*70}\n")
                
                for q_num, question_data in enumerate(initial_prompts, 1):
                    
                    question_id = question_data['id']
                    question_text = question_data['text']
                    short_label = prompt_manager.get_short_label(question_data)
                    
                    # Display image with question
                    display_text = f"Image: {20-len(selected_images)+idx+1}/20, Question {q_num}/{len(initial_prompts)}: {short_label}\n{question_text}\n\nPress: Y=Yes | N=No | S=Skip | Q=Quit"
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
                        "image_index": original_index,
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
                                                "image_index": original_index,
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
    CONTINENT = "Asia"
    CITY = "Tokio1"
    NUM_KEYFRAMES = 20  # Number of images to label
    OVERRIDE_EXISTING = True  # Set to True to re-label already labeled images
    SELECT_IMAGES = False
    PATCH_QUESTIONS = None

    """
    override_existing	patch_questions	        Behavior
        False	               None	        Normal incremental labeling
        False	               set(...)	    Patch only missing questions
        True	               None	        Full re-label all questions
        True	               set(...)	    Redo only those questions for all images"""

    # =======================================
    
    print(f"="*70)
    print(f"Ground Truth Labeling Tool - Hierarchical VQA")
    print(f"="*70)
    print(f"Preset: {PROMPT_PRESET}")
    print(f"City: {CITY}, {CONTINENT}")
    print(f"Target labels: ~{NUM_KEYFRAMES} images")
    print(f"="*70)
    
    input_directory = f"../data/{CONTINENT}/{CITY}"
    
    exit_code = create_ground_truth_labels(input_directory, PROMPT_PRESET, NUM_KEYFRAMES, OVERRIDE_EXISTING, False, PATCH_QUESTIONS)
    sys.exit(exit_code)

    # Rerun sittings question