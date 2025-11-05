from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import  os
import pyzed.sl as sl
from transformers import ViltProcessor, ViltForQuestionAnswering, InstructBlipProcessor, InstructBlipForConditionalGeneration
import cv2
from PIL import Image
import json
import torch
from inout.utils import progress_bar

BASE = "You are an expert at detecting pedestrian obstacles for people with low vision."
OBSTACLE_PROMPT = "Is there any obstacle blocking the user's presumed walking path?"
TRAFFIC_PROMPT = "Is there a crosswall or road in front?"
RED_LIGHT_PROMPT = "Is there a red light making the user to stop?"
ANOMALY_PROMPT = "Is there a contruction blocking the user's presumed walking path?"
CLEAN_PROMPT = "Does the path appear to be safe?"

ALL_PROMPTS = (OBSTACLE_PROMPT, TRAFFIC_PROMPT, RED_LIGHT_PROMPT, ANOMALY_PROMPT, CLEAN_PROMPT)

def main(svo_input_path,
         output_dir):

    # device + dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device} (dtype: {base_dtype})")

    # I/O
    enc_dir = os.path.join(output_dir, "encodings")
    os.makedirs(enc_dir, exist_ok=True)
    answers_path = os.path.join(output_dir, "answers.jsonl")   # answers.jsonl: one JSON per line {frame, answer, score(optional)}
    print(answers_path)
    # ZED init
    zed = sl.Camera()
    input_type = sl.InputType()
    init = sl.InitParameters(input_t=input_type)
    init.set_from_svo_file(svo_input_path)  # Set this path
    init.svo_real_time_mode = False # Don't convert in realtime
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init.depth_mode = sl.DEPTH_MODE.NEURAL  # Better quality
    init.enable_right_side_measure = False

    # Open the SVO file 
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

    # Start main loop
    print(f"VQA on SVO ({nb_frames} frames). Press Ctrl-C to stop.\n")
    try:
        with Path(answers_path).open("a", encoding="utf-8") as ans_f:
            while True:
                err = zed.grab(runtime)
                if err == sl.ERROR_CODE.SUCCESS:
                    frame = zed.get_svo_position()

                    if frame % FRAME_STRIDE == 0:
                        # get left image as PIL RGB
                        left = sl.Mat()
                        zed.retrieve_image(left, sl.VIEW.LEFT)
                        arr = left.get_data()
                        if arr.shape[2] == 4:
                            img_rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
                        else:
                            img_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(img_rgb)

                        # ----- cached encoding path -----
                        enc_path = os.path.join(enc_dir, f"frame_{frame:06d}.pt")
                        cached = False

                        # if os.path.exists(enc_path):
                        #     # load cached tensors 
                        #     encoding = torch.load(enc_path, weights_only=False).to(device=device)
                        #     cached = True
                        # else:
                        # build new encoding and save to CPU cache

                        # Run all the prompts about the semantic labels
                        for PROMPT in ALL_PROMPTS:

                            # Add the base
                            PROMPT = BASE + PROMPT
                            
                            # Run the encoding
                            encoding = processor(images=image, text=PROMPT, return_tensors="pt").to(device=device, dtype=base_dtype)

                            # Inference
                            with torch.no_grad():
                                if MODEL == "vilt":
                                    outputs = model(**encoding)
                                    logits = outputs.logits
                                    idx = logits.argmax(-1).item()
                                    answer_text = model.config.id2label[idx]
                                    prob = torch.softmax(logits, dim=-1)[0, idx].item()
                                    result_text = answer_text  # may not be strict JSON; keep raw text
                                    confidence = prob
                                else:  # blip2
                                    output_ids = model.generate(**encoding, max_new_tokens=256)
                                    result_text = processor.decode(output_ids[0], skip_special_tokens=True)
                                    result_text = result_text.split("Answer:")[-1].strip(" ,.;:")
                                    confidence = None

                            # ----- write result -----
                            ans_obj = {
                                "frame": int(frame),
                                "question": PROMPT,
                                "answer": result_text,
                                "confidence": confidence,
                                "cached_encoding": cached
                            }
                            ans_f.write(json.dumps(ans_obj) + "\n")
                            ans_f.flush()

                        # Save the last encoding (this needs to be fixed, so the encoding should be invariable to the prompt)
                        torch.save(encoding, enc_path)

                    # Progress
                    progress_bar((frame + 1) / max(1, nb_frames) * 100, 30)

                elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    progress_bar(100, 30)
                    sys.stdout.write("\nSVO end reached. Exiting.\n")
                    break
    finally:
        zed.close()
    return 0 

if __name__ == "__main__":
    
    MODEL = "vilt" # vilt or blip2
    FRAME_STRIDE = 100  # process every N-th frame

    seq = 17
    print(f"Processing sequence {seq}...")
    
    input_svo_path = f"../data/svo/IRI_{seq:02d}.svo2"
    output_directory = f"../data/vqa_outputs/IRI_{seq:02d}/{MODEL}/"
    
    main(input_svo_path, output_directory)













        
