
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import requests
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.parent

GLD_V2_COUNTRY_URL = "https://storage.googleapis.com/gld-v2/data/train/country/LT.json"
GLD_V2_LANDMARK_URL_TEMPLATE = "https://storage.googleapis.com/gld-v2/data/train/landmarks/{landmark_id}.json"


class GLDQuerier:
    def __init__(self, out_dir: str = "data", timeout: int = 30):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.landmarks_data = None
    
    def download_country_index(self, force: bool = False) -> bool:
        country_file = self.out_dir / "gldv2_lithuania.json"
        
        if country_file.exists() and not force:
            try:
                with open(country_file, 'r') as f:
                    self.landmarks_data = json.load(f)
                return True
            except Exception:
                pass
        
        try:
            response = requests.get(GLD_V2_COUNTRY_URL, timeout=self.timeout)
            response.raise_for_status()
            self.landmarks_data = response.json()
            
            with open(country_file, 'w') as f:
                json.dump(self.landmarks_data, f)
            
            print("Saved country index")
            return True
        except Exception:
            return False
    
    def get_landmark_image_count(self, landmark_id: int) -> int:
        try:
            url = GLD_V2_LANDMARK_URL_TEMPLATE.format(landmark_id=landmark_id)
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            landmark_data = response.json()
            
            images = landmark_data.get('images', [])
            return len(images) if isinstance(images, list) else 0
        except:
            return 0
    
    def query_all_landmarks(self) -> Dict[str, int]:
        if not self.landmarks_data:
            return {}
        
        results = {}
        for landmark_info in tqdm(self.landmarks_data, desc="Querying"):
            landmark_id = landmark_info.get('id')
            landmark_name = landmark_info.get('name', 'Unknown')
            
            if not landmark_id:
                continue
            
            image_count = self.get_landmark_image_count(landmark_id)
            results[landmark_name] = image_count
        
        return results


# Get top 100 landmarks with max images
def get_top_landmarks(image_counts: Dict[str, int], top_n: int = 100) -> Tuple[Dict[str, int], int]:
    if not image_counts:
        return {}, 0
    
    sorted_landmarks = sorted(image_counts.items(), key=lambda x: x[1], reverse=True)
    top_dict = {name: count for name, count in sorted_landmarks[:top_n]}
    min_count = min(top_dict.values()) if top_dict else 0
    
    return top_dict, min_count


def save_filtered_labels(filtered_labels: Dict[str, int], min_count: int, out_dir: str, original_count: int):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    sorted_labels = sorted(filtered_labels.items(), key=lambda x: x[1], reverse=True)
    
    csv_file = out_path / "gldv2_lithuania_labels_filtered.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["landmark_name", "image_count"])
        for name, count in sorted_labels:
            writer.writerow([name, count])
    
    stats = {
        "total_landmarks_queried": original_count,
        "top_n_selected": len(filtered_labels),
        "min_images_in_top_n": min_count,
        "max_images_in_top_n": max(filtered_labels.values()) if filtered_labels else 0,
        "avg_images_in_top_n": sum(filtered_labels.values()) / len(filtered_labels) if filtered_labels else 0,
        "total_images_for_training": sum(filtered_labels.values()),
    }
    
    stats_file = out_path / "gldv2_lithuania_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return csv_file, stats_file


def main():
    parser = argparse.ArgumentParser(description="Get top N Lithuanian landmarks from GLD v2")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "data"), help="Output directory")
    parser.add_argument("--force-download", action="store_true", help="Force re-download")
    parser.add_argument("--top-n", type=int, default=100, help="Number of top landmarks")
    
    args = parser.parse_args()
    
    querier = GLDQuerier(out_dir=args.out_dir)
    
    if not querier.download_country_index(force=args.force_download):
        print("Error: Could not load GLD v2 Lithuania data")
        sys.exit(1)
    
    image_counts = querier.query_all_landmarks()
    found = sum(1 for c in image_counts.values() if c > 0)
    
    if found == 0:
        print("Error: No landmarks found with images!")
        sys.exit(1)
    
    top_landmarks, min_count = get_top_landmarks(image_counts, top_n=args.top_n)
    
    if not top_landmarks:
        print("Error: No landmarks found!")
        sys.exit(1)
    
    save_filtered_labels(top_landmarks, min_count, args.out_dir, len(image_counts))
    print("Saved")


if __name__ == "__main__":
    main()
