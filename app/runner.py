from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
import pyzed.sl as sl

import cv2
from PIL import Image
import torch
from utils.utils import progress_bar

from core.promptManager import PromptManager
from core.vqaModel import VQAModel
from viz.viz_utils import create_display_frame, generate_event_map


def process_hierarchical_questions(image, prompt_manager, vqa_model, model_name):
    """
    Process questions hierarchically: Level 1 first, then conditional Level 2 based on answers.
    
    Returns:
        answers_dict: Dictionary of all answers for display
        result_objects: List of result objects for logging
    """
    answers_dict = {}
    result_objects = []
    
    # Reset answer history for new frame
    prompt_manager.reset_answer_history()
    
    # Step 1: Process all Level 1 questions
    initial_prompts = prompt_manager.get_initial_prompts()
    
    print(f"\n{'='*70}")
    print(f"LEVEL 1 QUESTIONS ({len(initial_prompts)} questions)")
    print(f"{'='*70}")
    
    for question_data in initial_prompts:
        question_id = question_data['id']
        question_prompt = question_data['text']
        full_prompt = prompt_manager.get_full_prompt(question_data)
        short_label = prompt_manager.get_short_label(question_data)
        
        # Process question
        try:
            answer_text, confidence = vqa_model.process_question(image, full_prompt)
        except Exception as e:
            print(f"Error processing question '{question_id}': {e}")
            try:
                answer_text, confidence = vqa_model.process_question(image, question_prompt)
            except Exception as e2:
                print(f"Error processing question '{question_id}' with basic prompt: {e2}")
                answer_text, confidence = None, None
        
        # Store answer
        if answer_text is not None:
            answers_dict[short_label] = {
                'answer': answer_text,
                'confidence': confidence
            }
            
            result_obj = {
                "question_id": question_id,
                "question": full_prompt,
                "answer": answer_text,
                "confidence": confidence,
                "model": model_name,
                "level": 1
            }
            result_objects.append(result_obj)
            
            # Print to console
            if confidence is not None:
                print(f"{short_label}: {answer_text} (conf: {confidence:.2f})")
            else:
                print(f"{short_label}: {answer_text}")
    
    # Step 2: Process Level 2 follow-up questions based on Level 1 answers
    print(f"\n{'='*70}")
    print(f"LEVEL 2 FOLLOW-UP QUESTIONS")                                                                               # LEVEL 3 NOT DONE
    print(f"{'='*70}")
    
    followup_count = 0
    for question_data in initial_prompts:
        question_id = question_data['id']
        short_label = prompt_manager.get_short_label(question_data)
        
        # Get the answer for this Level 1 question
        if short_label in answers_dict:
            answer = answers_dict[short_label]['answer']
            
            # Get follow-up questions
            followups = prompt_manager.get_followup_prompts(question_id, answer)
            
            if followups:
                print(f"\n--- Follow-ups for '{short_label}' (answered: {answer}) ---")
                followup_count += len(followups)
                
                for followup_q in followups:
                    followup_id = followup_q['id']
                    followup_prompt = followup_q['text']
                    full_followup = prompt_manager.get_full_prompt(followup_q)
                    followup_label = prompt_manager.get_short_label(followup_q)
                    
                    # Process follow-up question
                    try:
                        followup_answer, followup_conf = vqa_model.process_question(image, full_followup)
                    except Exception as e:
                        print(f"Error processing follow-up '{followup_id}': {e}")
                        try:
                            followup_answer, followup_conf = vqa_model.process_question(image, followup_prompt)
                        except Exception as e2:
                            print(f"Error processing follow-up '{followup_id}' with basic prompt: {e2}")
                            followup_answer, followup_conf = None, None
                    
                    # Store answer
                    if followup_answer is not None:
                        answers_dict[followup_label] = {
                            'answer': followup_answer,
                            'confidence': followup_conf
                        }
                        
                        result_obj = {
                            "question_id": followup_id,
                            "question": full_followup,
                            "answer": followup_answer,
                            "confidence": followup_conf,
                            "model": model_name,
                            "level": 2,
                            "parent_question": question_id
                        }
                        result_objects.append(result_obj)
                        
                        # Print to console
                        if followup_conf is not None:
                            print(f"  └─ {followup_label}: {followup_answer} (conf: {followup_conf:.2f})")
                        else:
                            print(f"  └─ {followup_label}: {followup_answer}")
    
    if followup_count == 0:
        print("No follow-up questions triggered (all Level 1 answers were negative)")
    else:
        print(f"\nProcessed {followup_count} follow-up questions")
    
    return answers_dict, result_objects


def fromSVO(svo_input_path, output_dir, model_name, prompt_preset, frame_stride):
    """Main processing function for SVO files."""
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device} (dtype: {base_dtype})")
    
    # I/O setup
    os.makedirs(output_dir, exist_ok=True)
    answers_path = os.path.join(output_dir, "answers.jsonl")
    print(f"Output: {answers_path}")
    
    # Load prompts
    prompt_manager = PromptManager(preset=prompt_preset)
    prompt_manager.print_hierarchy_info()
    
    # Initialize model
    vqa_model = VQAModel(model_name, device=device, dtype=base_dtype)
    
    # ZED Camera initialization
    zed = sl.Camera()
    input_type = sl.InputType()
    init = sl.InitParameters(input_t=input_type)
    init.set_from_svo_file(svo_input_path)
    init.svo_real_time_mode = False
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.enable_right_side_measure = False
    
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("ZED initialization failed")
        return 1
    
    runtime = sl.RuntimeParameters()
    nb_frames = zed.get_svo_number_of_frames()
    
    # Create window
    cv2.namedWindow("VQA Results", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VQA Results", 1280, 720)
    
    print(f"\nProcessing SVO: {nb_frames} frames")
    print(f"Processing every {frame_stride} frame(s)\n")
    
    try:
        with open(answers_path, "a", encoding="utf-8") as ans_f:
            while True:
                err = zed.grab(runtime)
                
                if err == sl.ERROR_CODE.SUCCESS:
                    frame = zed.get_svo_position()
                    
                    if frame % frame_stride == 0:
                        # Get image
                        left = sl.Mat()
                        zed.retrieve_image(left, sl.VIEW.LEFT)
                        arr = left.get_data()
                        
                        if arr.shape[2] == 4:
                            img_rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
                        else:
                            img_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                        
                        image = Image.fromarray(img_rgb)
                        
                        # Process hierarchically
                        print(f"\n{'='*70}")
                        print(f"FRAME {frame}")
                        print(f"{'='*70}")
                        
                        answers_dict, result_objects = process_hierarchical_questions(
                            image, prompt_manager, vqa_model, model_name
                        )
                        
                        # Write results to file
                        for result_obj in result_objects:
                            result_obj["frame"] = int(frame)
                            ans_f.write(json.dumps(result_obj) + "\n")
                        ans_f.flush()
                        
                        print(f"\n{'='*70}")
                        print(f"Processed {len(result_objects)} total questions")
                        print("Press any key to continue...")
                        
                        # Display frame
                        display_frame = create_display_frame(image, frame, answers_dict, model_name)
                        cv2.imshow("VQA Results", display_frame)
                        cv2.waitKey(0)
                    
                    # Progress bar
                    progress_bar((frame + 1) / max(1, nb_frames) * 100, 30)
                
                elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    progress_bar(100, 30)
                    print("\n\nSVO end reached. Processing complete!")
                    break
    
    finally:
        cv2.destroyAllWindows()
        zed.close()
    
    return 0


def fromImages(input_dir, output_dir, model_name, prompt_preset, num_keyframes=20, generate_map=False):
    """Main processing function for images from directory."""
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device} (dtype: {base_dtype})")
    
    # I/O setup
    image_dir = os.path.join(input_dir, "images")
    if generate_map:
        gps_csv_path = os.path.join(input_dir, "gps_positions.csv")
    os.makedirs(output_dir, exist_ok=True)
    answers_path = os.path.join(output_dir, "answers.jsonl")
    print(f"Output: {answers_path}")
    
    # Get list of image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))
        image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return 1
    
    print(f"Found {len(image_files)} images in {image_dir}")

    frame_stride = int(len(image_files)/num_keyframes)
    
    # Load prompts
    prompt_manager = PromptManager(preset=prompt_preset)
    
    if not generate_map:
        prompt_manager.print_hierarchy_info()
    
    # Initialize model
    vqa_model = VQAModel(model_name, device=device, dtype=base_dtype)
    
    # Create window
    if not generate_map:
        cv2.namedWindow("VQA Results", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("VQA Results", 1280, 720)
    
    print(f"\nProcessing {len(image_files)} images\n")
    
    try:
        with open(answers_path, "a", encoding="utf-8") as ans_f:
            for idx, image_path in enumerate(image_files):
                if idx % frame_stride == 0:
                    # Load image
                    try:
                        image = Image.open(image_path).convert('RGB')
                    except Exception as e:
                        print(f"Error loading {image_path}: {e}")
                        continue
                    
                    # Process hierarchically
                    if not generate_map:
                        print(f"\n{'='*70}")
                        print(f"IMAGE {idx+1}/{len(image_files)}: {image_path.name}")
                        print(f"{'='*70}")
                    
                    answers_dict, result_objects = process_hierarchical_questions(
                        image, prompt_manager, vqa_model, model_name
                    )
                    
                    # Write results to file
                    for result_obj in result_objects:
                        result_obj["image_path"] = str(image_path)
                        result_obj["image_name"] = image_path.name
                        result_obj["image_index"] = idx
                        ans_f.write(json.dumps(result_obj) + "\n")
                    ans_f.flush()
                    
                    if not generate_map:
                        print(f"\n{'='*70}")
                        print(f"Processed {len(result_objects)} total questions")
                        print("Press any key to continue (or 'q' to quit)...")
                        
                        # Display frame
                        display_frame = create_display_frame(image, idx, answers_dict, model_name)
                        cv2.imshow("VQA Results", display_frame)
                        key = cv2.waitKey(0)
                        
                        # Check for quit
                        if key == ord('q') or key == ord('Q'):
                            print("\nQuitting...")
                            break
                
                # Progress bar
                progress_bar((idx + 1) / max(1, len(image_files)) * 100, 30)
            
            progress_bar(100, 30)
            print("\n\nProcessing complete!")
    
    finally:
        cv2.destroyAllWindows()
    
    # Generate interactive map if requested
    if generate_map and gps_csv_path and os.path.exists(gps_csv_path):
        map_output_path = os.path.join(output_dir, "interactive_map.html")
        try:
            generate_event_map(gps_csv_path, answers_path, map_output_path, image_dir)
        except Exception as e:
            print(f"Error generating map: {e}")
    
    return 0


if __name__ == "__main__":

    """# From SVO
    # ============ CONFIGURATION ============
    MODEL = "instructblip"  # Options: vilt, blip2, blip2-large, llava, instructblip
    PROMPT_PRESET = "full_hierarchical"  # Options: level_1_only, crossing, stairs, construction, obstacle, crowding, vehicle, surface, visibility, full_hierarchical
    FRAME_STRIDE = 100  # Process every N-th frame
    # =======================================
    
    print(f"="*70)
    print(f"VQA Pedestrian Navigation System - Hierarchical Mode")
    print(f"="*70)
    print(f"Model: {MODEL}")
    print(f"Preset: {PROMPT_PRESET}")
    print(f"Sequence: {SEQUENCE}")
    print(f"="*70)
    
    SEQUENCE = 15 
    input_svo_path = f"../data/svo/IRI_{SEQUENCE:02d}.svo2"
    output_directory = f"../data/vqa_outputs/IRI_{SEQUENCE:02d}/{MODEL}_{PROMPT_PRESET}/"
    exit_code = fromSVO(input_svo_path, output_directory, MODEL, PROMPT_PRESET, FRAME_STRIDE)
    sys.exit(exit_code)"""

    # From Images 
    # ============ CONFIGURATION ============
    MODEL = "instructblip"  # Options: vilt, blip2, blip2-large, llava, instructblip
    PROMPT_PRESET = "full_hierarchical"  # Options: level_1_only, crossing, stairs, construction, obstacle, crowding, vehicle, surface, visibility, full_hierarchical
    CONTINENT = "America"
    CITY = "NewYork"
    # =======================================
    
    print(f"="*70)
    print(f"VQA Pedestrian Navigation System - Hierarchical Mode")
    print(f"="*70)
    print(f"Model: {MODEL}")
    print(f"Preset: {PROMPT_PRESET}")
    print(f"City: {CITY}, {CONTINENT}")
    print(f"="*70)

    image_dir = f"../data/{CONTINENT}/{CITY}"
    output_dir = f"../data/vqa_outputs/{CITY}/{MODEL}_{PROMPT_PRESET}/"
    
    exit_code = fromImages(image_dir, output_dir, MODEL, PROMPT_PRESET, num_keyframes=20, generate_map=True)
    sys.exit(exit_code)