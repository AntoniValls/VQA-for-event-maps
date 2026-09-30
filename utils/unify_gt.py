"""
This codes reads all individual ground truth files (ground_truth_labels.jsonl) 
from the data folder and merges them into a single file (combined_ground_truth.jsonl).
 It also prints the total number of unique records (questions) found across all files.
 """
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import DATA_DIR

def merge_ground_truth(root_dir, output_file):
    root_path = Path(root_dir)
    all_records = []
    
    # Recursively find all ground_truth_labels.jsonl files
    # This matches the structure: data/*/*/ground_truth_labels.jsonl
    jsonl_files = sorted(root_path.glob("*/*/ground_truth_labels.jsonl"))
    
    print(f"Found {len(jsonl_files)} sequence files. Starting merge...")

    for file_path in jsonl_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))

    # Write the combined data to a new unique file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for record in all_records:
            out_f.write(json.dumps(record) + '\n')

    print("--- Results ---")
    print(f"Total unique records (questions) found: {len(all_records)}")
    print(f"Combined file saved to: {output_file}")

if __name__ == "__main__":
    # Point this to your 'data' folder
    merge_ground_truth(root_dir=DATA_DIR, output_file=DATA_DIR / "all_GT" / "all_ground_truth.jsonl")