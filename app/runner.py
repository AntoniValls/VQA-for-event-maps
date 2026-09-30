from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import os
import json

import cv2
from PIL import Image
import torch
from inout.utils import progress_bar

from core.paths import MODELS, list_sequences, record_path, sequence_dir
from core.promptManager import PromptManager
from core.vqaModel import VQAModel
from core.eval import evaluate_model_performance
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
    
    # Level 1
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
    print(f"LEVEL 2 FOLLOW-UP QUESTIONS")                                                                               
    print(f"{'='*70}")
    
    followup_count = 0
    followups_array = []
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
                followups_array.append(followups)    
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

    # Step 3: Process Level 3 follow-up questions based on Level 2 answers
    print(f"\n{'='*70}")
    print(f"LEVEL 3 FOLLOW-UP QUESTIONS")                                                                              
    print(f"{'='*70}")
    
    followup_count = 0
    for followups2 in followups_array:
        for question_data in followups2:
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
                                "level": 3,
                                "parent_question": question_id
                            }
                            result_objects.append(result_obj)
                            
                            # Print to console
                            if followup_conf is not None:
                                print(f"  └─ {followup_label}: {followup_answer} (conf: {followup_conf:.2f})")
                            else:
                                print(f"  └─ {followup_label}: {followup_answer}")
        
    if followup_count == 0:
        print("No follow-up questions triggered (all Level 2 answers were negative)")
    else:
        print(f"\nProcessed {followup_count} follow-up questions")
    
    return answers_dict, result_objects

_loaded_model = {}

def get_model(model_name, device, dtype):
    """Load a model once and reuse it for all sequences (keeps only one model in memory)."""
    if _loaded_model.get("name") != model_name:
        _loaded_model.clear()
        if device == "cuda":
            torch.cuda.empty_cache()
        _loaded_model["name"] = model_name
        _loaded_model["model"] = VQAModel(model_name, device=device, dtype=dtype)
    return _loaded_model["model"]

def fromImages(input_dir, model_name, prompt_preset, num_keyframes=20, generate_map=False, evaluate=False):
    """Main processing function for images from directory."""
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device} (dtype: {base_dtype})")
    
    # I/O setup
    image_dir = os.path.join(input_dir, "images_selected")
    if generate_map:
        gps_csv_path = os.path.join(input_dir, "gps_positions.csv")
    answers_folder = os.path.join(input_dir,f"results/{model_name}")
    os.makedirs(answers_folder, exist_ok=True)
    answers_path = os.path.join(answers_folder, "answers.jsonl")
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

    frame_stride = max(1, len(image_files) // num_keyframes)
    
    # Load prompts
    prompt_manager = PromptManager(preset=prompt_preset)
    
    if not generate_map:
        prompt_manager.print_hierarchy_info()
    
    # Initialize model
    vqa_model = get_model(model_name, device, base_dtype)
    
    # # Create window
    # if not generate_map:
    #     cv2.namedWindow("VQA Results", cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow("VQA Results", 1280, 720)
    
    print(f"\nProcessing {len(image_files)} images\n")
    
    try:
        with open(answers_path, "w", encoding="utf-8") as ans_f:
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
                        result_obj["image_path"] = record_path(image_path)
                        result_obj["image_name"] = image_path.name
                        result_obj["image_index"] = idx
                        ans_f.write(json.dumps(result_obj) + "\n")
                    ans_f.flush()
                    
                    # if not generate_map:
                    #     print(f"\n{'='*70}")
                    #     print(f"Processed {len(result_objects)} total questions")
                    #     print("Press any key to continue (or 'q' to quit)...")
                        
                    #     # Display frame
                    #     display_frame = create_display_frame(image, idx, answers_dict, model_name)
                    #     cv2.imshow("VQA Results", display_frame)
                    #     key = cv2.waitKey(0)
                        
                    #     # Check for quit
                    #     if key == ord('q') or key == ord('Q'):
                    #         print("\nQuitting...")
                    #         break
                
                # Progress bar
                progress_bar((idx + 1) / max(1, len(image_files)) * 100, 30)
            
            progress_bar(100, 30)
            print("\n\nProcessing complete!")
    
    finally:
        cv2.destroyAllWindows()
    
    # Generate interactive map if requested
    if generate_map and gps_csv_path and os.path.exists(gps_csv_path):
        map_output_path = os.path.join(answers_folder, "interactive_map.html")
        try:
            generate_event_map(gps_csv_path, answers_path, map_output_path, image_dir, show=True)
        except Exception as e:
            print(f"Error generating map: {e}")
    
    # Evaluate with the GT if requested
    if evaluate:
        gt_path = os.path.join(input_dir, "ground_truth_labels.jsonl")
        evaluate_model_performance(answers_path, gt_path)
    
    return 0


if __name__ == "__main__":
    # Runs the VQA models over the annotated sequences found in the data folder.
    #   python app/runner.py                                          -> all sequences, all models
    #   python app/runner.py --models qwen-vl --continent Asia --city Tokio1
    parser = argparse.ArgumentParser(description="Run the hierarchical VQA models over the dataset")
    parser.add_argument("--models", nargs="+", default=MODELS, help=f"Models to run (default: {' '.join(MODELS)})")
    parser.add_argument("--continent", help="Only this continent (default: all)")
    parser.add_argument("--city", help="Only this sequence folder (default: all)")
    parser.add_argument("--preset", default="full_hierarchical", help="Question preset (default: full_hierarchical)")
    parser.add_argument("--num-keyframes", type=int, default=20)
    parser.add_argument("--map", action="store_true", help="Also generate the interactive risk event map")
    parser.add_argument("--no-eval", action="store_true", help="Skip the evaluation against the GT")
    args = parser.parse_args()

    sequences = list_sequences(args.continent, args.city)
    if not sequences:
        sys.exit("No annotated sequences found (need data/<Continent>/<City>/ground_truth_labels.jsonl)")

    errors = []
    for model in args.models:
        for continent, city in sequences:
            print(f"="*70)
            print(f"VQA Pedestrian Navigation System - Hierarchical Mode")
            print(f"="*70)
            print(f"Model: {model}")
            print(f"Preset: {args.preset}")
            print(f"City: {city}, {continent}")
            print(f"="*70)

            try:
                fromImages(str(sequence_dir(continent, city)), model, args.preset,
                           num_keyframes=args.num_keyframes, generate_map=args.map, evaluate=not args.no_eval)
            except Exception as e:
                errors.append(f"ERROR: --- {model} | {continent}/{city}: {e}")
                print(errors[-1])

    if errors:
        print("\nThese are the errors we got:")
        for error in errors:
            print(error)
