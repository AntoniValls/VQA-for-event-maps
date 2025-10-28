from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import  os
import pyzed.sl as sl
from transformers import ViltProcessor, ViltForQuestionAnswering, Blip2Processor, Blip2ForConditionalGeneration
import cv2
from PIL import Image
import json
import torch
from inout.utils import progress_bar

"""
Simple file that allows us to run a VQA model on a .svo file
and save the answer and the image embedding
"""

QUESTION = "Describe any obstacles in the scene."
MODEL = "blip2" # vilt or blip2

# Prompt template helps models stay in VQA mode
PROMPT = f"Question: {QUESTION} Answer:"

def main(svo_input_path,
         output_dir):

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

    # Prepare the output directory
    enc_dir = os.path.join(output_dir, "encodings")
    os.makedirs(enc_dir, exist_ok=True)
    answers_path = os.path.join(output_dir, "answers.jsonl")   # answers.jsonl: one JSON per line {frame, answer, score(optional)}

    # Initialize the variables
    nb_frames = zed.get_svo_number_of_frames()
    left_image = sl.Mat()

    # Initialize the encoder and VQA  model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if MODEL == "vilt":
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa", dtype=torch.float16)
    elif MODEL == "blip2":
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", dtype=torch.float16)
    else:
        raise ValueError(f"Unknown model: {MODEL}. Choose 'vilt' or 'blip2'.")
    model.to(device)

    # Start main loop
    sys.stdout.write(f"VQA on SVO ({nb_frames} frames)... Use Ctrl-C to interrupt conversion.\n")

    with Path(answers_path).open("a", encoding="utf-8") as ans_f:
        while True:
            err = zed.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                frame = zed.get_svo_position()

                # Process every 100th frame
                if frame % 100 == 0:
                    zed.retrieve_image(left_image, sl.VIEW.LEFT)

                    # sl.Mat --> PIL RGB
                    image = Image.fromarray(cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGR2RGB))

                    # Encode the image
                    encoding = processor(images=image, text=PROMPT, return_tensors="pt").to(device, torch.float16)

                    # Save the encodings
                    torch.save(encoding, os.path.join(enc_dir, f"frame_{frame:06d}.pt"))

                    # VQA
                    with torch.no_grad():
                        if MODEL == "vilt":
                            outputs = model(**encoding)
                            logits = outputs.logits
                            probs = torch.softmax(logits, dim=-1)
                            idx = logits.argmax(-1).item()
                            answer = model.config.id2label[idx]
                            confidence = probs[0, idx].item()
                        
                        elif MODEL == "blip2":
                            output = model.generate(**encoding)
                            answer = processor.decode(output[0], skip_special_tokens=True)
                            confidence = None  # BLIP2 does not provide confidence scores directly
                    
                    # Write answer line
                    ans_obj = {
                        "frame": int(frame),
                        "question": QUESTION,
                        "answer": answer,
                        "confidence": confidence
                    }
                    ans_f.write(json.dumps(ans_obj) + "\n")
                    ans_f.flush()

                
                progress_bar((frame + 1) / max(1, nb_frames) * 100, 30)

            elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                progress_bar(100, 30)
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break

    zed.close()
    return 0 

if __name__ == "__main__":
    seq = 0
    print(f"Processing sequence {seq}...")
    
    input_svo_path = f"../data/svo/IRI_{seq:02d}.svo2"
    output_directory = f"../data/vqa_outputs/IRI_{seq:02d}/{MODEL}/"
    
    main(input_svo_path, output_directory)













        
