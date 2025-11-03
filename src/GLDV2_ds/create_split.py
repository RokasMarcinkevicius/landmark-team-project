
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.parent


class SplitCreator:
    def __init__(self, out_dir: str = None):
        if out_dir is None:
            out_dir = str(SCRIPT_DIR / "data")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
    
    def load_metadata(self, metadata_file: str) -> Dict[str, List[Dict]]:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Create train/val/test split .txt files
    def create_split_files(self, metadata: Dict[str, List[Dict]], train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[int, int, int]:
        
        train_file = self.out_dir / "train.txt"
        val_file = self.out_dir / "val.txt"
        test_file = self.out_dir / "test.txt"
        
        train_lines = []
        val_lines = []
        test_lines = []
        
        test_ratio = 1.0 - train_ratio - val_ratio
        
        for landmark_name, images in tqdm(metadata.items(), desc="Creating split"):
            image_ids = [img.get('image_id', '') for img in images]
            
            if len(image_ids) < 2:
                train_lines.extend([f"{landmark_name},{img_id}" for img_id in image_ids])
                continue
            
            train_ids, temp_ids = train_test_split(
                image_ids, 
                test_size=(1 - train_ratio), 
                random_state=42
            )
            
            val_size = val_ratio / (val_ratio + test_ratio)
            val_ids, test_ids = train_test_split(
                temp_ids,
                test_size=(1 - val_size),
                random_state=42
            )
            
            train_lines.extend([f"{landmark_name},{img_id}" for img_id in train_ids])
            val_lines.extend([f"{landmark_name},{img_id}" for img_id in val_ids])
            test_lines.extend([f"{landmark_name},{img_id}" for img_id in test_ids])
        

        with open(train_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_lines))
        
        with open(val_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_lines))
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_lines))
        
        return len(train_lines), len(val_lines), len(test_lines)


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test split files")
    parser.add_argument("--metadata", default=str(SCRIPT_DIR / "data" / "metadata.json"), help="Input metadata.json")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "data"), help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio (default: 0.15)")
    
    args = parser.parse_args()
    
    creator = SplitCreator(out_dir=args.out_dir)
    
    metadata = creator.load_metadata(args.metadata)
    
    if not metadata:
        print("Error: No metadata loaded!")
        sys.exit(1)
    
    train_count, val_count, test_count = creator.create_split_files(
        metadata,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    total = train_count + val_count + test_count
    print("Saved")
    print(f"Train: {train_count} ({train_count/total*100:.1f}%)")
    print(f"Val:   {val_count} ({val_count/total*100:.1f}%)")
    print(f"Test:  {test_count} ({test_count/total*100:.1f}%)")


if __name__ == "__main__":
    main()
