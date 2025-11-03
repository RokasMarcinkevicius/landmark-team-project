
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent


# Load metadata.json with all image URLs
def load_metadata() -> Dict[str, List[Dict]]:
    metadata_file = PROJECT_ROOT / "data" / "metadata.json"
    with open(metadata_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_split(split_name: str) -> List[Tuple[str, str]]:
    split_file = PROJECT_ROOT / "data" / f"{split_name}.txt"
    data = []
    with open(split_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.rsplit(',', 1)
                if len(parts) == 2:
                    data.append((parts[0], parts[1]))
    return data


# Create mapping from landmark_name to class_id
def get_landmark_to_class_id(metadata: Dict[str, List[Dict]]) -> Dict[str, int]:
    mapping = {}
    for class_id, landmark_name in enumerate(sorted(metadata.keys())):
        mapping[landmark_name] = class_id
    return mapping


# Download image from URL
def download_image(url: str, timeout: int = 10) -> np.ndarray:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return np.array(img)
    except Exception:
        return None


# Generator (image, label) pairs
def create_image_generator(split_name: str, metadata: Dict[str, List[Dict]], 
                           landmark_to_class: Dict[str, int], img_size: int = 224):
    split_data = load_split(split_name)
    
    for landmark_name, image_id in split_data:
        # Find image URL from metadata
        if landmark_name not in metadata:
            continue
        
        image_data = None
        for img in metadata[landmark_name]:
            if img.get('image_id') == image_id:
                image_data = img
                break
        
        if image_data is None:
            continue
        
        url = image_data.get('url')
        if not url:
            continue
        
        # Download and process image
        img_array = download_image(url)
        if img_array is None:
            continue
        
        # Resize to img_size
        img_pil = Image.fromarray(img_array)
        img_pil = img_pil.resize((img_size, img_size), Image.BILINEAR)
        img_array = np.array(img_pil, dtype=np.float32) / 255.0
        
        # Get class label
        class_id = landmark_to_class[landmark_name]
        label = tf.one_hot(class_id, depth=len(landmark_to_class))
        
        yield img_array, label


# Create TensorFlow datasets for train or validation
def get_tf_datasets(split_name: str = "train", img_size: int = 224, batch: int = 32, seed: int = 42):
    metadata = load_metadata()
    landmark_to_class = get_landmark_to_class_id(metadata)
    num_classes = len(landmark_to_class)
    
    ds = tf.data.Dataset.from_generator(
        lambda: create_image_generator(split_name, metadata, landmark_to_class, img_size),
        output_signature=(
            tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
        )
    )
    
    ds = ds.batch(batch)
    if split_name == "train":
        ds = ds.shuffle(buffer_size=100, seed=seed)
    
    return ds, list(landmark_to_class.keys())


# Create both train and val TensorFlow datasets
def get_tf_datasets_pair(img_size: int = 224, batch: int = 32, seed: int = 42):
    train_ds, class_names = get_tf_datasets("train", img_size, batch, seed)
    val_ds, _ = get_tf_datasets("val", img_size, batch, seed)
    
    return train_ds, val_ds
