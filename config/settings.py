"""
All configurable parameters used

"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# -------- Top-level groups -----------------------------------------------------

@dataclass
class Recording:
    base_run_dir: str = "../data/recordings"                         
    log_filename: str = "run.log"  
    gps_filename: str = "gps_data.csv"     
    svo_filename: str = "recording.svo2"                                          

@dataclass
class GPS:
    # WebSocket / adapter
    ws_port: int = 8085                                         # http port

@dataclass
class ZED:
    sdk_verbose: int = 1
    coordinate_units: str = "METER"                             # sl.UNIT.METER
    coordinate_system: str = "RIGHT_HANDED_Z_UP_X_FWD"          # sl.COORDINATE_SYSTEM
    camera_resolution: str = "HD720"                            # sl.RESOLUTION
    camera_fps: int = 60                                        # frames per second

@dataclass
class Config:
    recording: Recording = field(default_factory=Recording)
    gps: GPS = field(default_factory=GPS)
    zed: ZED = field(default_factory=ZED)

# single import for the whole app
CFG = Config()
