"""
Script for GPS recording and utilities.
"""
import csv
import logging
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict
from dataclasses import dataclass
import websocket
import warnings
import numpy as np
import matplotlib.pyplot as plt
import time
import pyzed.sl as sl
import math
from collections import deque 

warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass
class GPSCoordinate:
    """
    Single GPS fix (WGS84).

    Attributes
    ----------
    timestamp : datetime
        Host-receive time for the fix.
    frame : Optional[int]
        Current SVO frame index at the time the fix was received.
    latitude, longitude : float
        Degrees (WGS84).
    accuracy : float
        Horizontal accuracy (meters). Semantics depend on source device.
    time : int
        Device-provided GPS time (milliseconds), if available.
    speed : float
        Device speed estimate (m/s), if available.
    """
    timestamp: datetime
    latitude: float
    longitude: float
    accuracy: float
    time: int
    speed: float
    frame: Optional[int] = None  

    def to_dict(self) -> dict:
        """Serialize to a JSON/CSV-friendly dict."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'frame': self.frame,  
            'latitude': self.latitude,
            'longitude': self.longitude,
            'accuracy': self.accuracy,
            'time': self.time,
            'speed': self.speed
        }
    
class GPSLogger:
    """ Thread-safe logger for GPS coordinate"""

    def __init__(self, csv_file: str = "gps_log.csv"):
        self.csv_file = Path(csv_file)
        self.gps_data: List[GPSCoordinate] = []
        self._lock = threading.Lock()
        self.coordinates: List[GPSCoordinate] = []
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Ensure the directory for the CSV file exists."""
        self.csv_file.parent.mkdir(parents=True, exist_ok=True)

    def log_gps(self, coord: GPSCoordinate) -> None:
        """Log a new GPS coordinate."""
        with self._lock:
            self.gps_data.append(coord)

    def save_to_csv(self) -> None:
        """Save logged GPS data to a CSV file."""
        with self._lock:
            gps_list = list(self.gps_data)

        with open(self.csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(
                file,
                fieldnames=['timestamp', 'frame', 'latitude', 'longitude', 'accuracy', 'time', 'speed']  # <-- NEW field 'frame'
            )
            writer.writeheader()
            for coord in gps_list:
                writer.writerow(coord.to_dict())
        logging.info(f"Saved {len(gps_list)} GPS records to {self.csv_file}")


class GPSWebSocketClient:
    """GPS WebSocket client that feeds a GPSLogger from a mobile/edge server."""

    def __init__(self, data_logger: GPSLogger, port: int = 8080, frame_counter: Optional[object] = None) -> None:
        self.ip_address = self._read_ip_address()
        self.gps_url = f"ws://{self.ip_address}:{port}/gps" if self.ip_address else None
        self.data_logger = data_logger
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.running = False
        self.frame_counter = frame_counter

    def _read_ip_address(self) -> Optional[str]:
        """
        Read the target IP address from ../../config/ip_address.txt.

        Returns
        -------
        str | None
            The IP address string, or None on failure.
        """
        ip_file = Path("../config/smartphone_ip_address.txt")
        if not ip_file.exists():
            logging.error("IP address file not found!")
            return None

        try:
            with open(ip_file, 'r', encoding='utf-8') as file:
                ip = file.read().strip()
                if ip:
                    return ip
            
        except Exception as e:
            logging.error(f"Failed to read IP address: {e}")

        return None

    def on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        """Handle incoming JSON messages; store GPSCoordinate to the logger."""
        try:
            data = json.loads(message)
            if 'latitude' not in data or 'longitude' not in data:
                return

            # Capture the current frame index at the instant this GPS fix arrives
            current_frame = None
            if self.frame_counter is not None:
                try:
                    current_frame = self.frame_counter.get()
                except Exception:
                    current_frame = None

            coord = GPSCoordinate(
                timestamp=datetime.now(),
                latitude=float(data['latitude']),
                longitude=float(data['longitude']),
                accuracy=float(data.get('accuracy', 0)),
                time=int(data.get('time', 0)),
                speed=float(data.get('speed', 0)) if 'speed' in data else 0.0,
                frame=current_frame  # <-- NEW
            )
            self.data_logger.log_gps(coord) # Log GPS with frame tag
        except Exception as e:
            logging.error(f"GPS message error: {e}")

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        """Mark running and request last known location."""
        logging.info("GPS WebSocket connected successfully")
        self.running = True
        try:
            ws.send("getLastKnowLocation")
            logging.info("Sent initial GPS location request")
        except Exception as e:
            logging.error(f"Failed to send initial GPS command: {e}")

    def on_close(self, ws: websocket.WebSocketApp, *args: Any) -> None:
        """Closed by server or client; mark as not running."""
        self.running = False
        logging.info("GPS WebSocket closed")

    def on_error(self, ws: websocket.WebSocketApp, error: Any) -> None:
        """Handle WebSocket errors."""
        logging.error(f"GPS WebSocket error: {error}")

    def connect(self) -> None:
        """Start the WebSocket client in a daemon thread."""
        if not self.gps_url:
            logging.error("Cannot connect GPS WebSocket: missing URL/IP.")
            return

        self.ws = websocket.WebSocketApp(
            self.gps_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.ws_thread.start()

    def disconnect(self) -> None:
        """Close the WebSocket and mark as not running."""
        if self.ws:
            self.ws.close()
        self.running = False
