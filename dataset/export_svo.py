import os
import sys
import cv2
import json
import numpy as np
import pyzed.sl as sl
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.paths import DATA_DIR

def progress_bar(percent_done, bar_length=50):
    #Display a progress bar
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %i%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()

def export_images(svo_input_path, output_dir, max_frames=None):

    # ZED init
    zed = sl.Camera()
    input_type = sl.InputType()
    init = sl.InitParameters(input_t=input_type)
    init.set_from_svo_file(svo_input_path)  # ← Set this path
    init.svo_real_time_mode = False # Don't convert in realtime

    # Open the SVO file 
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("ZED initialization failed")
        exit(1)
    
    runtime = sl.RuntimeParameters()

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    
    left_image = sl.Mat()

    # Initialize variables
    old_imu_timestamp = 0
    nb_frames = zed.get_svo_number_of_frames()

    # Start SVO conversion
    sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")

    # Define the stride for frame extraction
    if max_frames is not None:
        stride = max(1, nb_frames // max_frames)
    else:
        stride = 1

    while True:
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:

            frame = zed.get_svo_position()

            if frame % stride != 0:
                continue  # Skip frames based on the defined stride

            # Retrieve images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            left_img_file = os.path.join(output_dir, ("%s.png" % str(frame).zfill(6)))
            cv2.imwrite(str(left_img_file),left_image.get_data())

            # Display progress  
            progress_bar((frame + 1) / nb_frames * 100, 30)

        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            progress_bar(100 , 30)
            sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
            break

    zed.close()
    return 0

if __name__ == "__main__":

    sequences = [f"{i:02}" for i in range(0, 23)]
    print("Available sequences:", sequences)
    for seq in sequences:
        print(f"Processing sequence {seq}...")
        input_svo_path = str(DATA_DIR / "Europe" / "Barcelona" / "svo" / f"IRI_{seq}.svo2")
        output_directory = str(DATA_DIR / "Europe" / seq / "images")
        export_images(input_svo_path, output_directory, max_frames=100)
    

