
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.parent

GLD_V2_LANDMARK_URL_TEMPLATE = "https://storage.googleapis.com/gld-v2/data/train/landmarks/{landmark_id}.json"
GLD_V2_COUNTRY_FILE = lambda: SCRIPT_DIR / "data" / "gldv2_lithuania.json"


class MetadataFetcher:
    def __init__(self, out_dir: str = None, timeout: int = 30):
        if out_dir is None:
            out_dir = str(SCRIPT_DIR / "data")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.country_data = self._load_country_data()
        self.landmark_id_map = self._build_landmark_map()
    
    def _load_country_data(self) -> List[Dict]:
        country_file = Path(GLD_V2_COUNTRY_FILE())
        if not country_file.exists():
            return []
        
        try:
            with open(country_file, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    
    def _build_landmark_map(self) -> Dict[str, int]:
        mapping = {}
        for landmark in self.country_data:
            landmark_id = landmark.get('id')
            landmark_name = landmark.get('name', '')
            if landmark_id and landmark_name:
                mapping[landmark_name] = landmark_id
        return mapping
    
    #Read top 100 landmarks from filtered .csv
    def read_filtered_landmarks(self, csv_file: str, top_n: int) -> List[Tuple[str, int]]:
        landmarks = []
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]
                for i, line in enumerate(lines):
                    if i >= top_n:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.rsplit(',', 1)
                    if len(parts) == 2:
                        name = parts[0].strip()
                        try:
                            count = int(parts[1].strip())
                            if name and count > 0:
                                landmarks.append((name, count))
                        except ValueError:
                            continue
        except Exception as e:
            return []
        
        return landmarks
    
    #Download landmark .json and extract image metadata
    def fetch_landmark_images(self, landmark_name: str) -> List[Dict]:
        
        landmark_id = self.landmark_id_map.get(landmark_name)
        if not landmark_id:
            return []
        
        try:
            url = GLD_V2_LANDMARK_URL_TEMPLATE.format(landmark_id=landmark_id)
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            landmark_data = response.json()
            
            images = landmark_data.get('images', [])
            if not isinstance(images, list):
                return []
            
            result = []
            for img in images:
                result.append({
                    'image_id': img.get('id', ''),
                    'url': img.get('url', ''),
                })
            
            return result
        except Exception:
            return []
    
    #Apply class balancing
    def balance_classes(self, landmark_images: Dict[str, List[Dict]], images_per_class: Optional[int] = None) -> Dict[str, List[Dict]]:
        if not landmark_images:
            return {}
        
        total_images = sum(len(imgs) for imgs in landmark_images.values())
        num_classes = len(landmark_images)
        
        if images_per_class is None or images_per_class == "balanced":
            target_per_class = total_images // num_classes
        else:
            target_per_class = int(images_per_class)
        
        balanced = {}
        for landmark_name, images in landmark_images.items():
            if len(images) > target_per_class:
                step = len(images) / target_per_class
                sampled = [images[int(i * step)] for i in range(target_per_class)]
                balanced[landmark_name] = sampled
            else:
                balanced[landmark_name] = images
        
        return balanced
    
    def save_metadata_json(self, landmark_images: Dict[str, List[Dict]], out_file: str) -> Tuple[Path, int]:
        
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        total_images = 0
        data = {}
        for landmark_name, images in landmark_images.items():
            data[landmark_name] = images
            total_images += len(images)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return out_path, total_images


def main():
    parser = argparse.ArgumentParser(description="Fetch metadata for top N landmarks")
    parser.add_argument("--filtered-csv", default=str(SCRIPT_DIR / "data" / "gldv2_lithuania_labels_filtered.csv"), help="Filtered landmarks CSV")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "data"), help="Output directory")
    parser.add_argument("--top-n", type=int, default=53, help="Number of top landmarks")
    parser.add_argument("--images-per-class", default="balanced", help="'balanced' or integer N")
    
    args = parser.parse_args()
    
    fetcher = MetadataFetcher(out_dir=args.out_dir)
    
    top_landmarks = fetcher.read_filtered_landmarks(args.filtered_csv, args.top_n)
    
    if not top_landmarks:
        print("Error: No landmarks loaded!")
        sys.exit(1)
    
    landmark_images = {}
    
    for landmark_name, _ in tqdm(top_landmarks, desc="Fetching"):
        images = fetcher.fetch_landmark_images(landmark_name)
        if images:
            landmark_images[landmark_name] = images
    
    if not landmark_images:
        print("Error: No images fetched!")
        sys.exit(1)
    
    if args.images_per_class == "balanced":
        balanced_images = fetcher.balance_classes(landmark_images, None)
    else:
        try:
            balanced_images = fetcher.balance_classes(landmark_images, int(args.images_per_class))
        except ValueError:
            print(f"Error: Invalid --images-per-class value: {args.images_per_class}")
            sys.exit(1)
    
    metadata_json = args.out_dir + "/metadata.json"
    json_file, total_images = fetcher.save_metadata_json(balanced_images, metadata_json)
    print("Saved")

if __name__ == "__main__":
    main()
