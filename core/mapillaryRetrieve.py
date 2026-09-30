"""
Downloads a Mapillary sequence into data/<Continent>/<City>/ (images/, metadata.json, gps_positions.csv).

    python core/mapillaryRetrieve.py --continent Asia --city Hanoi --sequence <SEQUENCE_ID>
"""
import argparse
import sys
import requests
import json
from pathlib import Path
from tqdm import tqdm
import time
from PIL import Image
from io import BytesIO
import math

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import CONTINENTS, get_setting, sequence_dir

def extract_front_view(image_bytes, compass_angle=0, fov=90):
    """
    Extract front view from a 360° equirectangular image.
    
    Args:
        image_bytes: Raw image bytes
        compass_angle: Compass angle in degrees (0-360)
        fov: Field of view in degrees (default 90)
    
    Returns:
        PIL Image object of the front view
    """
    img = Image.open(BytesIO(image_bytes))
    width, height = img.size
    
    # Check if it's a 360 image (width should be ~2x height for equirectangular)
    aspect_ratio = width / height
    if aspect_ratio < 1.8:  # Not a 360 image
        return img
    
    # Calculate the center longitude based on compass angle
    # Compass angle of 0 = North, 90 = East, etc.
    center_lon = compass_angle
    
    # Calculate the horizontal slice to extract
    # For equirectangular projection, longitude maps linearly to x-axis
    fov_rad = math.radians(fov)
    
    # Calculate pixel range for the FOV
    pixels_per_degree = width / 360
    half_fov_pixels = int((fov / 2) * pixels_per_degree)
    
    # Center pixel based on compass angle
    center_pixel = int((center_lon / 360) * width)
    
    # Calculate left and right bounds
    left = (center_pixel - half_fov_pixels) % width
    right = (center_pixel + half_fov_pixels) % width
    
    # Handle wrapping around the image
    if left < right:
        # Simple case: no wrapping
        front_view = img.crop((left, 0, right, height))
    else:
        # Wrapping case: need to stitch two parts
        right_part = img.crop((left, 0, width, height))
        left_part = img.crop((0, 0, right, height))
        
        # Create new image and paste parts
        front_view = Image.new('RGB', (right_part.width + left_part.width, height))
        front_view.paste(right_part, (0, 0))
        front_view.paste(left_part, (right_part.width, 0))
    
    return front_view

def mapillary_retrieve(mly_key, seq, output_folder):
     
    images_folder = output_folder / 'images'
    images_folder.mkdir(exist_ok=True)

    # Step 1: Get all image IDs from the sequence
    print("Fetching image IDs from sequence...")
    url = f'https://graph.mapillary.com/image_ids?access_token={mly_key}&sequence_id={seq}'
    response = requests.get(url)

    if response.status_code != 200:
        raise RuntimeError(f"Error fetching sequence {seq}: HTTP {response.status_code} {response.text[:200]}")

    data = response.json()
    image_ids = [obj['id'] for obj in data['data']]
    print(f"Found {len(image_ids)} images in sequence")

    # Step 2: Download images and collect metadata
    metadata = []
    detections = {}

    for idx, image_id in enumerate(tqdm(image_ids[::2], desc="Downloading images")):
        if idx <= 600: # Limit
            # Get image metadata (including GPS and camera type)
            meta_url = f'https://graph.mapillary.com/{image_id}?access_token={mly_key}&fields=id,thumb_2048_url,geometry,captured_at,compass_angle,camera_type,is_pano,camera_parameters,width,height'            
            meta_response = requests.get(meta_url)
            
            if meta_response.status_code == 200:
                image_data = meta_response.json()
                
                # Extract GPS coordinates
                geometry = image_data.get('geometry', {})
                coordinates = geometry.get('coordinates', [None, None])
                lon, lat = coordinates[0], coordinates[1]
                
                compass_angle = image_data.get('compass_angle', 0)
                is_pano = image_data.get('is_pano', False)
                camera_type = image_data.get('camera_type', 'unknown')
                camera_parameters = image_data.get('camera_parameters', 'unknown')
                heigh = image_data.get('height', 'unknown')
                width = image_data.get('width', 'unknown')

                # Download the image
                image_url = image_data.get('thumb_2048_url')
                if image_url:
                    img_response = requests.get(image_url)
                    if img_response.status_code == 200:
                        # Process image based on whether it's 360 or not
                        if is_pano or camera_type == 'spherical':
                            print("ATENTION: images are 360s")
                            # Extract front view from 360 image
                            front_view_img = extract_front_view(
                                img_response.content, 
                                compass_angle=compass_angle,
                                fov=90  # 90 degree field of view
                            )
                            image_type = '360_front'
                        else:
                            # Regular image, just open it
                            front_view_img = Image.open(BytesIO(img_response.content))
                            image_type = 'regular'
                        
                        # Save image with sequence number
                        image_filename = f"{idx:04d}_{image_id}.jpg"
                        image_path = images_folder / image_filename
                        front_view_img.save(image_path, 'JPEG', quality=95)
                        
                        # Store metadata
                        metadata.append({
                            'image_id': image_id,
                            'seq_id': seq,
                            'filename': image_filename,
                            'latitude': lat,
                            'longitude': lon,
                            'captured_at': image_data.get('captured_at'),
                            'compass_angle': compass_angle,
                            'is_pano': is_pano,
                            'camera_type': camera_type,
                            'image_type': image_type,
                            'camera_parameters': camera_parameters,
                            'width': width,
                            'heigh': heigh
                        })
            
            # Rate limiting - be nice to the API
            time.sleep(0.1)

    # Step 3: Save metadata to JSON files
    print("\nSaving metadata...")

    # Save GPS positions
    gps_data = [{
        'image_id': item['image_id'],
        'filename': item['filename'],
        'latitude': item['latitude'],
        'longitude': item['longitude'],
        'captured_at': item['captured_at'],
        'compass_angle': item['compass_angle'],
        'is_pano': item['is_pano'],
        'image_type': item['image_type'],
        'camera_parameters': item['camera_parameters'],
        'width': item['width'],
        'heigh': item['heigh']
    } for item in metadata]

    # Save complete metadata
    with open(output_folder / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create a simple CSV for easy viewing
    with open(output_folder / 'gps_positions.csv', 'w') as f:
        f.write('filename,latitude,longitude,captured_at,compass_angle,is_pano,image_type\n')
        for item in gps_data:
            f.write(f"{item['filename']},{item['latitude']},{item['longitude']},{item['captured_at']},{item['compass_angle']},{item['is_pano']},{item['image_type']}\n")

    print(f"\nDownload complete!")
    print(f"Images saved to: {images_folder}")
    print(f"Metadata saved to: {output_folder / 'metadata.json'}")
    print(f"CSV saved to: {output_folder / 'gps_positions.csv'}")

    # Print summary
    pano_count = sum(1 for item in metadata if item['image_type'] == '360_front')
    regular_count = sum(1 for item in metadata if item['image_type'] == 'regular')
    print(f"\nProcessed {pano_count} 360° images (extracted front view)")
    print(f"Processed {regular_count} regular images")

    return

def load_mapillary_token():
    """Read MAPILLARY_TOKEN from the environment or from the (git-ignored) .env file."""
    token = get_setting("MAPILLARY_TOKEN")
    if not token:
        raise RuntimeError(
            "MAPILLARY_TOKEN not found. Copy .env.example to .env and paste your "
            "Mapillary client token there (see README, 'Mapillary access token')."
        )
    return token

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Mapillary sequence")
    parser.add_argument("--continent", required=True, choices=CONTINENTS)
    parser.add_argument("--city", required=True, help="Folder name, CamelCase without spaces (e.g. BuenosAires, Tokio2)")
    parser.add_argument("--sequence", required=True, help="Mapillary sequence ID")
    args = parser.parse_args()

    mly_key = load_mapillary_token()

    output_folder = sequence_dir(args.continent, args.city)
    if (output_folder / "images").is_dir() and any((output_folder / "images").iterdir()):
        sys.exit(f"{output_folder}/images already exists and is not empty. Delete it or choose another --city.")

    # Create output directories
    output_folder.mkdir(parents=True, exist_ok=True)

    mapillary_retrieve(mly_key, args.sequence, output_folder)
