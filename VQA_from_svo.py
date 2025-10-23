"""
Simple file that allows us to run a VQA model on a .svo file
and save the answer and the image embedding
"""
import sys, os
from pathlib import Path
import pyzed.sl as sl
from transformers import ViltProcessor, ViltForQuestionAnswering
import cv2
from PIL import Image
import json
import torch

QUESTION = "Is there something blocking the sidewalk?"

def progress_bar(percent_done, bar_length=50):
    #Display a progress bar
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %i%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()

def main(svo_input_path,
         output_dir):
    
    # ZED init
    zed = sl.Camera()
    input_type = sl.InputType()
    init = sl.InitParameters(input_t=input_type)
    init.set_from_svo_file(svo_input_path)  # ← Set this path
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
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    
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
                    encoding = processor(images=image, text=QUESTION, return_tensors="pt")

                    # Save the encodings
                    torch.save(encoding, os.path.join(enc_dir, f"frame_{frame:06d}.pt"))

                    # VQA
                    with torch.no_grad():
                        outputs = model(**encoding)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=-1)
                        idx = logits.argmax(-1).item()
                        answer = model.config.id2label[idx]
                        confidence = probs[0, idx].item()

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
    
    input_svo_path = f"./data/svo/IRI_{seq:02d}.svo2"
    output_directory = f"./data/vqa_outputs/IRI_{seq:02d}"
    
    main(input_svo_path, output_directory)













        
