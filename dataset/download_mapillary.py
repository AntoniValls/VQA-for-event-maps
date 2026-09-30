"""
Downloads a Mapillary sequence into data/<Continent>/<City>/ (images/ + metadata.json with the per-image positions).
360° sequences are rejected. Positions use Mapillary's computed (SfM-refined) geometry when available.

    python dataset/download_mapillary.py --continent Asia --city Hanoi --sequence <SEQUENCE_ID>
    python dataset/download_mapillary.py --continent Asia --city Hanoi --image <IMAGE_ID>   # any image of the sequence

Sequences not captured on foot (e.g. from a car) are rejected unless --allow-not-on-foot is given.
The web viewer calls a sequence a "capture": the capture key is the sequence ID.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.paths import CONTINENTS, get_setting, sequence_dir

# 360° imagery is not accepted: only regular (perspective / fisheye) cameras.
PANORAMIC_CAMERA_TYPES = {"spherical", "equirectangular"}

IMAGE_FIELDS = ("id,thumb_2048_url,computed_geometry,geometry,captured_at,computed_compass_angle,compass_angle,"
                "camera_type,is_pano,camera_parameters,width,height,on_foot")

def is_panoramic(image_data):
    return bool(image_data.get('is_pano')) or image_data.get('camera_type') in PANORAMIC_CAMERA_TYPES

def get_position(image_data):
    """
    Position and heading of an image. Uses Mapillary's computed (SfM-refined) values,
    falling back to the raw device GPS/compass when the image has not been processed.
    """
    geometry = image_data.get('computed_geometry')
    source = 'computed'
    if not geometry:
        geometry = image_data.get('geometry', {})
        source = 'original'
    lon, lat = geometry.get('coordinates', [None, None])

    compass_angle = image_data.get('computed_compass_angle')
    if compass_angle is None:
        compass_angle = image_data.get('compass_angle', 0)
    return lat, lon, compass_angle, source

def api_get(url, retries=5, backoff=2.0):
    """
    GET with retries: Mapillary often answers with transient errors (HTTP 5xx / 429,
    "Service temporarily unavailable") that succeed a few seconds later.
    """
    response = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200 or (response.status_code < 500 and response.status_code != 429):
                return response
        except requests.RequestException as e:
            print(f"  Request error: {e}")
        if attempt < retries - 1:
            wait = backoff * 2 ** attempt
            status = response.status_code if response is not None else "no response"
            print(f"  Mapillary error ({status}), retrying in {wait:.0f}s ({attempt + 1}/{retries - 1})...")
            time.sleep(wait)
    return response

def get_image_data(mly_key, image_id):
    url = f'https://graph.mapillary.com/{image_id}?access_token={mly_key}&fields={IMAGE_FIELDS}'
    response = api_get(url)
    if response is None or response.status_code != 200:
        return None
    return response.json()

def get_sequence_of_image(mly_key, image_id):
    """Sequence ID (the "capture key" in the web viewer) that an image belongs to."""
    response = api_get(f'https://graph.mapillary.com/{image_id}?access_token={mly_key}&fields=id,sequence')
    if response is None or response.status_code != 200:
        detail = f"HTTP {response.status_code} {response.text[:200]}" if response is not None else "no response"
        raise RuntimeError(f"Could not read image {image_id}: {detail}")
    seq = response.json().get('sequence')
    if not seq:
        raise RuntimeError(f"Image {image_id} does not belong to any sequence")
    return seq

def get_sequence_image_ids(mly_key, seq):
    """Image IDs of a sequence in capture order."""
    response = api_get(f'https://graph.mapillary.com/image_ids?access_token={mly_key}&sequence_id={seq}')
    if response is not None and response.status_code == 200:
        return [obj['id'] for obj in response.json()['data']]

    # Fallback: image search endpoint (not ordered -> sort by capture time)
    print("  image_ids endpoint failed, trying the image search endpoint...")
    fallback = api_get(f'https://graph.mapillary.com/images?access_token={mly_key}'
                       f'&sequence_ids={seq}&fields=id,captured_at&limit=2000')
    if fallback is not None and fallback.status_code == 200:
        images = sorted(fallback.json()['data'], key=lambda x: x.get('captured_at', 0))
        return [obj['id'] for obj in images]

    last = fallback if fallback is not None else response
    detail = f"HTTP {last.status_code} {last.text[:200]}" if last is not None else "no response"
    raise RuntimeError(f"Error fetching sequence {seq}: {detail}. "
                       "If the error is transient, just try again in a few minutes.")

def mapillary_retrieve(mly_key, seq, output_folder, max_images=601, allow_not_on_foot=False):
     
    images_folder = output_folder / 'images'

    # Step 1: Get all image IDs from the sequence
    print("Fetching image IDs from sequence...")
    image_ids = get_sequence_image_ids(mly_key, seq)
    print(f"Found {len(image_ids)} images in sequence")
    if not image_ids:
        raise RuntimeError(f"Sequence {seq} has no images")

    # Reject 360° sequences before downloading anything
    first = get_image_data(mly_key, image_ids[0])
    if first is None:
        raise RuntimeError(f"Could not read the metadata of image {image_ids[0]}")
    if is_panoramic(first):
        raise RuntimeError(f"Sequence {seq} is 360° (camera_type={first.get('camera_type')}). "
                           "360° sequences are not accepted, choose another one.")

    # Pedestrian point of view: on_foot is True / False, or None when Mapillary doesn't know
    on_foot = first.get('on_foot')
    if on_foot is False and not allow_not_on_foot:
        raise RuntimeError(f"Sequence {seq} was not captured on foot (on_foot=False, probably from a vehicle). "
                           "Choose a pedestrian sequence, or use --allow-not-on-foot if you are sure.")
    if on_foot is None:
        print("WARNING: Mapillary doesn't know whether this sequence was captured on foot. "
              "Check in the images that it is a pedestrian point of view.")

    images_folder.mkdir(parents=True, exist_ok=True)

    # Step 2: Download every second image (up to max_images) and collect metadata
    metadata = []
    skipped_pano = 0
    skipped_error = 0

    for idx, image_id in enumerate(tqdm(image_ids[::2][:max_images], desc="Downloading images")):
        image_data = get_image_data(mly_key, image_id)
        image_url = image_data.get('thumb_2048_url') if image_data else None
        if not image_url:
            skipped_error += 1
            continue

        # Mixed sequences: skip any 360° image
        if is_panoramic(image_data):
            skipped_pano += 1
            continue

        img_response = api_get(image_url)
        if img_response is None or img_response.status_code != 200:
            skipped_error += 1
            continue

        lat, lon, compass_angle, position_source = get_position(image_data)

        # Save image with sequence number
        image_filename = f"{idx:04d}_{image_id}.jpg"
        Image.open(BytesIO(img_response.content)).convert('RGB').save(images_folder / image_filename, 'JPEG', quality=95)

        # Store metadata (both computed and original positions are kept)
        metadata.append({
            'image_id': image_id,
            'seq_id': seq,
            'filename': image_filename,
            'latitude': lat,
            'longitude': lon,
            'position_source': position_source,
            'computed_geometry': image_data.get('computed_geometry'),
            'geometry': image_data.get('geometry'),
            'captured_at': image_data.get('captured_at'),
            'compass_angle': compass_angle,
            'computed_compass_angle': image_data.get('computed_compass_angle'),
            'original_compass_angle': image_data.get('compass_angle'),
            'is_pano': image_data.get('is_pano', False),
            'on_foot': image_data.get('on_foot'),
            'camera_type': image_data.get('camera_type', 'unknown'),
            'image_type': 'regular',
            'camera_parameters': image_data.get('camera_parameters', 'unknown'),
            'width': image_data.get('width', 'unknown'),
            'height': image_data.get('height', 'unknown')
        })

        # Rate limiting - be nice to the API
        time.sleep(0.1)

    # Step 3: Save metadata
    print("\nSaving metadata...")

    # Save complete metadata (the only per-sequence metadata file; also used by the event maps)
    with open(output_folder / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDownload complete!")
    print(f"Images saved to: {images_folder}")
    print(f"Metadata saved to: {output_folder / 'metadata.json'}")

    # Print summary
    n_computed = sum(1 for item in metadata if item['position_source'] == 'computed')
    print(f"\nDownloaded {len(metadata)} images "
          f"({n_computed} with computed position, {len(metadata) - n_computed} with original GPS)")
    if skipped_pano:
        print(f"Skipped {skipped_pano} 360° images")
    if skipped_error:
        print(f"Skipped {skipped_error} images that could not be downloaded")

    return

def load_mapillary_token():
    """Read MAPILLARY_TOKEN from the environment or from the (git-ignored) .env file."""
    token = get_setting("MAPILLARY_TOKEN")
    if token and "your_client" in token:
        raise RuntimeError(
            "MAPILLARY_TOKEN in .env is still the example placeholder. Replace it with your own "
            "Client Token from https://www.mapillary.com/dashboard/developers (README, 'Mapillary access token'). "
            "Note: Mapillary answers invalid tokens with a misleading 'Service temporarily unavailable'.")
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence", help='Mapillary sequence ID ("capture key" in the web viewer)')
    source.add_argument("--image", help="ID of any image of the sequence (pKey=... in the viewer URL)")
    parser.add_argument("--allow-not-on-foot", action="store_true",
                        help="Accept sequences that Mapillary marks as not captured on foot")
    args = parser.parse_args()

    try:
        mly_key = load_mapillary_token()
    except RuntimeError as e:
        sys.exit(f"ERROR: {e}")

    output_folder = sequence_dir(args.continent, args.city)
    if (output_folder / "images").is_dir() and any((output_folder / "images").iterdir()):
        sys.exit(f"{output_folder}/images already exists and is not empty. Delete it or choose another --city.")

    try:
        seq = args.sequence
        if args.image:
            print(f"Looking up the sequence of image {args.image}...")
            seq = get_sequence_of_image(mly_key, args.image)
            print(f"Image {args.image} belongs to sequence {seq}")
        mapillary_retrieve(mly_key, seq, output_folder, allow_not_on_foot=args.allow_not_on_foot)
    except RuntimeError as e:
        sys.exit(f"ERROR: {e}")
