from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import pyzed.sl as sl
from transformers import ViltProcessor, ViltForQuestionAnswering, InstructBlipProcessor, InstructBlipForConditionalGeneration
import cv2
from PIL import Image
import json
import torch
from inout.utils import progress_bar
import numpy as np

BASE = "You are an expert at detecting pedestrian obstacles for people with low vision."
OBSTACLE_PROMPT = "Is there any obstacle blocking the user's presumed walking path?"
TRAFFIC_PROMPT = "Is there a crosswalk or road in front?"
RED_LIGHT_PROMPT = "Is there a red light making the user to stop?"
ANOMALY_PROMPT = "Is there a contruction blocking the user's presumed walking path?"
CLEAN_PROMPT = "Does the path appear to be safe?"

ALL_PROMPTS = (OBSTACLE_PROMPT, TRAFFIC_PROMPT, RED_LIGHT_PROMPT, ANOMALY_PROMPT, CLEAN_PROMPT)

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
    
    return text_height + baseline + 5  # Return height for next line

def create_display_frame(image, frame_number, answers_dict):
    """Create a display frame with the image and answers overlay."""
    # Convert PIL Image to OpenCV format if needed
    if isinstance(image, Image.Image):
        img_display = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img_display = image.copy()
    
    # Add frame number
    draw_text_with_background(img_display, f"Frame: {frame_number}", (10, 30), 
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
        
        # Choose color based on answer (customize as needed)
        if 'yes' in answer.lower():
            if prompt_short == "Safe Path":
                text_color = (0, 255, 0) 
            else:
                text_color = (0, 0, 155)
        elif 'no' in answer.lower():
            if prompt_short != "Safe Path":
                text_color = (0, 255, 0) 
            else:
                text_color = (0, 0, 155)
        else:
            text_color = (255, 255, 255)  # White for other answers
        
        height = draw_text_with_background(img_display, text, (10, y_offset), 
                                          font_scale=0.6, font_thickness=1,
                                          text_color=text_color, bg_color=(0, 0, 0))
        y_offset += height
    
    return img_display

def main(svo_input_path, output_dir):
    # device + dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device} (dtype: {base_dtype})")

    # I/O
    enc_dir = os.path.join(output_dir, "encodings")
    os.makedirs(enc_dir, exist_ok=True)
    answers_path = os.path.join(output_dir, "answers.jsonl")
    print(answers_path)
    
    # ZED init
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
        exit(1)
    
    runtime = sl.RuntimeParameters()
    nb_frames = zed.get_svo_number_of_frames()
    
    # Load model + processor
    if MODEL == "vilt":
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        torch_dtype = base_dtype if device == "cuda" else torch.float32
        model = ViltForQuestionAnswering.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa", dtype=torch_dtype
        )
    elif MODEL == "blip2":
        if device != "cuda":
            print("BLIP2 generally requires CUDA for practical inference.")
            return 1
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b", use_fast=True)
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", dtype=base_dtype)
    else:
        print(f"Unknown model: {MODEL}. Choose 'vilt' or 'blip2'.")
        return 1
    
    model.to(device)
    model.eval()

    # Create window for display
    cv2.namedWindow("VQA Results", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VQA Results", 1280, 720)

    print(f"\nVQA on SVO ({nb_frames} frames).")
    print("Processing every {FRAME_STRIDE} frames.")
    print("When a stride frame is processed, click on the image or press any key to continue.\n")
    
    try:
        with Path(answers_path).open("a", encoding="utf-8") as ans_f:
            while True:
                err = zed.grab(runtime)
                if err == sl.ERROR_CODE.SUCCESS:
                    frame = zed.get_svo_position()

                    if frame % FRAME_STRIDE == 0:
                        # Get left image
                        left = sl.Mat()
                        zed.retrieve_image(left, sl.VIEW.LEFT)
                        arr = left.get_data()
                        if arr.shape[2] == 4:
                            img_rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
                        else:
                            img_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(img_rgb)

                        answers_dict = {}
                        enc_path = os.path.join(enc_dir, f"frame_{frame:06d}.pt")
                        cached = False

                        # Run all the prompts
                        for PROMPT in ALL_PROMPTS:
                            # Create short label for display
                            if "obstacle" in PROMPT.lower():
                                prompt_short = "Obstacle"
                            elif "crosswall" in PROMPT.lower() or "road" in PROMPT.lower():
                                prompt_short = "Traffic"
                            elif "red light" in PROMPT.lower():
                                prompt_short = "Red Light"
                            elif "contruction" in PROMPT.lower():
                                prompt_short = "Anomaly"
                            elif "safe" in PROMPT.lower():
                                prompt_short = "Safe Path"
                            else:
                                prompt_short = "Unknown"
                            
                            full_prompt = BASE + PROMPT
                            encoding = processor(images=image, text=full_prompt, return_tensors="pt").to(device=device, dtype=base_dtype)

                            # Inference
                            with torch.no_grad():
                                if MODEL == "vilt":
                                    outputs = model(**encoding)
                                    logits = outputs.logits
                                    idx = logits.argmax(-1).item()
                                    answer_text = model.config.id2label[idx]
                                    prob = torch.softmax(logits, dim=-1)[0, idx].item()
                                    result_text = answer_text
                                    confidence = prob
                                else:  # blip2
                                    output_ids = model.generate(**encoding, max_new_tokens=256)
                                    result_text = processor.decode(output_ids[0], skip_special_tokens=True)
                                    result_text = result_text.split("Answer:")[-1].strip(" ,.;:")
                                    confidence = None

                            # Store answer for display
                            answers_dict[prompt_short] = {
                                'answer': result_text,
                                'confidence': confidence
                            }

                            # Write result
                            ans_obj = {
                                "frame": int(frame),
                                "question": full_prompt,
                                "answer": result_text,
                                "confidence": confidence,
                                "cached_encoding": cached
                            }
                            ans_f.write(json.dumps(ans_obj) + "\n")
                            ans_f.flush()

                        # Save encoding
                        torch.save(encoding, enc_path)
                        
                        # Create and display frame with answers
                        display_frame = create_display_frame(image, frame, answers_dict)
                        cv2.imshow("VQA Results", display_frame)
                        
                        # Print answers to console
                        print(f"\n{'='*60}")
                        print(f"FRAME {frame} - PROCESSED")
                        print(f"{'='*60}")
                        for prompt_short, answer_data in answers_dict.items():
                            answer = answer_data['answer']
                            confidence = answer_data.get('confidence')
                            if confidence is not None:
                                print(f"  {prompt_short:12s}: {answer} (confidence: {confidence:.2f})")
                            else:
                                print(f"  {prompt_short:12s}: {answer}")
                        print(f"{'='*60}")
                        print("Click on image or press any key to continue...")

                        # Wait for user to click or press a key
                        cv2.waitKey(0)

                    # Progress
                    progress_bar((frame + 1) / max(1, nb_frames) * 100, 30)

                elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    progress_bar(100, 30)
                    sys.stdout.write("\nSVO end reached. Exiting.\n")
                    break
                    
    finally:
        cv2.destroyAllWindows()
        zed.close()
    return 0

if __name__ == "__main__":
    MODEL = "vilt"  # vilt or blip2
    FRAME_STRIDE = 100  # process every N-th frame

    seq = 10
    print(f"Processing sequence {seq}...")
    
    input_svo_path = f"../data/svo/IRI_{seq:02d}.svo2"
    output_directory = f"../data/vqa_outputs/IRI_{seq:02d}/{MODEL}/"
    
    main(input_svo_path, output_directory)