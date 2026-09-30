"""
Runs the VQA models over the annotated sequences and writes results/<model>/answers.jsonl.
Evaluation and event maps are separate steps (evaluation/evaluate.py, evaluation/event_map.py).

    python vqa/run.py                                             # all sequences, all models
    python vqa/run.py --models qwen-vl --continent Asia --city Tokio1
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from common.paths import MODELS, list_sequences, record_path, sequence_dir
from common.utils import progress_bar
from vqa.models import VQAModel
from vqa.prompt_manager import PromptManager

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def ask(vqa_model, image, prompt_manager, question):
    """Ask one question (with the base context). Falls back to the bare question if the model rejects it."""
    try:
        return vqa_model.process_question(image, prompt_manager.get_full_prompt(question))
    except Exception as e:
        print(f"Error processing question '{question['id']}': {e}")
    try:
        return vqa_model.process_question(image, question['text'])
    except Exception as e:
        print(f"Error processing question '{question['id']}' with basic prompt: {e}")
    return None, None


def process_hierarchical_questions(image, prompt_manager, vqa_model, model_name):
    """
    Ask the question hierarchy level by level: all Level-1 questions, then the follow-ups
    of every 'yes' answer (Level 2), then the follow-ups of those (Level 3).

    Returns:
        List of result objects (one per answered question) for answers.jsonl
    """
    prompt_manager.reset_answer_history()
    results = []

    # (question, parent_id) pairs to ask at the current level
    current = [(q, None) for q in prompt_manager.get_initial_prompts()]
    level = 1

    while current:
        print(f"\n{'='*70}\nLEVEL {level} QUESTIONS ({len(current)})\n{'='*70}")
        next_level = []

        for question, parent_id in current:
            answer, confidence = ask(vqa_model, image, prompt_manager, question)
            if answer is None:
                continue

            label = prompt_manager.get_short_label(question)
            conf_txt = f" (conf: {confidence:.2f})" if confidence is not None else ""
            print(f"{'  └─ ' if level > 1 else ''}{label}: {answer}{conf_txt}")

            result = {
                "question_id": question['id'],
                "question": prompt_manager.get_full_prompt(question),
                "answer": answer,
                "confidence": confidence,
                "model": model_name,
                "level": level,
            }
            if parent_id is not None:
                result["parent_question"] = parent_id
            results.append(result)

            for followup in prompt_manager.get_followup_prompts(question['id'], answer):
                next_level.append((followup, question['id']))

        current = next_level
        level += 1

    return results


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


def run_sequence(seq_dir, model_name, prompt_preset="full_hierarchical", num_keyframes=20):
    """Run one model over the keyframes (images_selected/) of one sequence."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device} (dtype: {dtype})")

    image_dir = Path(seq_dir) / "images_selected"
    answers_folder = Path(seq_dir) / "results" / model_name
    answers_folder.mkdir(parents=True, exist_ok=True)
    answers_path = answers_folder / "answers.jsonl"
    print(f"Output: {answers_path}")

    image_files = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_files:
        raise FileNotFoundError(f"No images found in {image_dir}")
    print(f"Found {len(image_files)} images in {image_dir}")

    frame_stride = max(1, len(image_files) // num_keyframes)

    prompt_manager = PromptManager(preset=prompt_preset)
    prompt_manager.print_hierarchy_info()
    vqa_model = get_model(model_name, device, dtype)

    with open(answers_path, "w", encoding="utf-8") as ans_f:
        for idx, image_path in enumerate(image_files):
            if idx % frame_stride == 0:
                image = Image.open(image_path).convert('RGB')
                print(f"\n{'='*70}\nIMAGE {idx+1}/{len(image_files)}: {image_path.name}\n{'='*70}")

                for result in process_hierarchical_questions(image, prompt_manager, vqa_model, model_name):
                    result["image_path"] = record_path(image_path)
                    result["image_name"] = image_path.name
                    result["image_index"] = idx
                    ans_f.write(json.dumps(result) + "\n")
                ans_f.flush()

            progress_bar((idx + 1) / len(image_files) * 100, 30)

    print("\n\nProcessing complete!")
    return answers_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hierarchical VQA models over the dataset")
    parser.add_argument("--models", nargs="+", default=MODELS, help=f"Models to run (default: {' '.join(MODELS)})")
    parser.add_argument("--continent", help="Only this continent (default: all)")
    parser.add_argument("--city", help="Only this sequence folder (default: all)")
    parser.add_argument("--preset", default="full_hierarchical", help="Question preset (default: full_hierarchical)")
    parser.add_argument("--num-keyframes", type=int, default=20)
    args = parser.parse_args()

    sequences = list_sequences(args.continent, args.city)
    if not sequences:
        sys.exit("No annotated sequences found (need data/<Continent>/<City>/ground_truth_labels.jsonl)")

    errors = []
    for model in args.models:
        for continent, city in sequences:
            print(f"{'='*70}\nVQA Pedestrian Navigation System - Hierarchical Mode")
            print(f"Model: {model} | Preset: {args.preset} | {continent}/{city}\n{'='*70}")
            try:
                run_sequence(sequence_dir(continent, city), model, args.preset, args.num_keyframes)
            except Exception as e:
                errors.append(f"ERROR: --- {model} | {continent}/{city}: {e}")
                print(errors[-1])

    if errors:
        print("\nThese are the errors we got:")
        for error in errors:
            print(error)
    else:
        print("\nDone. Evaluate with: python evaluation/evaluate.py")
