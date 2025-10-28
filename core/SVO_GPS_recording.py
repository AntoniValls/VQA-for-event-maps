"""
This code allows us to record a SVO file using a connected ZED camera
while it also saves the GPS data from a connected GPS module.
"""
from pathlib import Path
import sys, os
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import logging
import pyzed.sl as sl
import time
from typing import Optional

from config.settings import CFG
from GPS_utils import GPSCoordinate, GPSLogger, GPSWebSocketClient
from inout.utils import create_run_dir

# Simple thread-safe frame counter ----------
import threading

class FrameCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0

    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    def get(self) -> int:
        with self._lock:
            return self._value

def record():

    # --------------------- Initial setup ---------------------
    run_dir = create_run_dir(CFG.recording.base_run_dir) 

    # Setup logging
    log_path = os.path.join(run_dir, CFG.recording.log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path)
        ]
    )
    logging.info(f"Run directory: {run_dir}")
    
    # Initialize GPS logger
    gps_logger = GPSLogger(os.path.join(run_dir, CFG.recording.gps_filename))
    
    # Create a shared frame counter so GPS records can tag their frame
    frame_counter = FrameCounter()

    # Start GPS WebSocket client (Thread inside)
    gps_client = GPSWebSocketClient(gps_logger, port=CFG.gps.ws_port, frame_counter=frame_counter) 
    gps_client.connect()
    time.sleep(5)   # Wait for GPS connection
    if not gps_client.running:
        logging.error("Failed to connect to GPS")
        return
    
    # Initialize ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.sdk_verbose = 1
    init_params.coordinate_units  = getattr(sl.UNIT, CFG.zed.coordinate_units)
    init_params.coordinate_system = getattr(sl.COORDINATE_SYSTEM, CFG.zed.coordinate_system)
    init_params.camera_resolution = getattr(sl.RESOLUTION, CFG.zed.camera_resolution)
    init_params.camera_fps = CFG.zed.camera_fps                          
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        logging.error(f"Camera Open failed: {status}")
        gps_client.disconnect()
        return

    # Set SVO recording parameters
    svo_params = sl.RecordingParameters(CFG.recording.svo_filename, sl.SVO_COMPRESSION_MODE.H265)
    err = zed.enable_recording(svo_params)
    if err != sl.ERROR_CODE.SUCCESS:
        logging.error(f"Error starting SVO recording: {err}")
        zed.close()
        gps_client.disconnect()
        return

    runtime = sl.RuntimeParameters()
    print("SVO is Recording, use Ctrl-C to stop.") 
    frames_recorded = 0
    latest_gps: Optional[GPSCoordinate] = None      # cache last known GPS 


    # --------------------- Recording loop --------------------- 
    try:
        while True:
            if zed.grab(runtime) <= sl.ERROR_CODE.SUCCESS:  # new frame acquired
                current_frame = frame_counter.increment()
                print("Frame count: " + str(current_frame), end="\r")
            
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
    
    finally:
        # Save the gps data log
        gps_logger.save_to_csv()

        # Stop the GPS and SVO recording and close the camera
        zed.close()
        gps_client.disconnect()
    
    return

if __name__ == "__main__":
    record()
