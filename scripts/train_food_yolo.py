"""
Smart Food Waste Estimator - YOLO Food Segmentation Trainer
===========================================================
This script downloads the EduardoPacheco/FoodSeg103 dataset, converts its 
semantic segmentations into YOLO instance segmentation polygons, and trains 
a YOLO11 segmentation model.

Usage:
  pixi run python scripts/train_food_yolo.py --subset-size 100 --epochs 3
"""

import os
# Resolve OpenMP conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import yaml
import argparse
import numpy as np
from datasets import load_dataset
from ultralytics import YOLO

# 103 classes mapping (excluding background ID 0)
CLASS_NAMES = {
    1: "candy", 2: "egg tart", 3: "french fries", 4: "chocolate", 5: "biscuit",
    6: "popcorn", 7: "pudding", 8: "ice cream", 9: "cheese butter", 10: "cake",
    11: "wine", 12: "milkshake", 13: "coffee", 14: "juice", 15: "milk",
    16: "tea", 17: "almond", 18: "red beans", 19: "cashew", 20: "dried cranberries",
    21: "soy", 22: "walnut", 23: "peanut", 24: "egg", 25: "apple",
    26: "date", 27: "apricot", 28: "avocado", 29: "banana", 30: "strawberry",
    31: "cherry", 32: "blueberry", 33: "raspberry", 34: "mango", 35: "olives",
    36: "peach", 37: "lemon", 38: "pear", 39: "fig", 40: "pineapple",
    41: "grape", 42: "kiwi", 43: "melon", 44: "orange", 45: "watermelon",
    46: "steak", 47: "pork", 48: "chicken duck", 49: "sausage", 50: "fried meat",
    51: "lamb", 52: "sauce", 53: "crab", 54: "fish", 55: "shellfish",
    56: "shrimp", 57: "soup", 58: "bread", 59: "corn", 60: "hamburg",
    61: "pizza", 62: " hanamaki baozi", 63: "wonton dumplings", 64: "pasta", 65: "noodles",
    66: "rice", 67: "pie", 68: "tofu", 69: "eggplant", 70: "potato",
    71: "garlic", 72: "cauliflower", 73: "tomato", 74: "kelp", 75: "seaweed",
    76: "spring onion", 77: "rape", 78: "ginger", 79: "okra", 80: "lettuce",
    81: "pumpkin", 82: "cucumber", 83: "white radish", 84: "carrot", 85: "asparagus",
    86: "bamboo shoots", 87: "broccoli", 88: "celery stick", 89: "cilantro mint", 90: "snow peas",
    91: " cabbage", 92: "bean sprouts", 93: "onion", 94: "pepper", 95: "green beans",
    96: "French beans", 97: "king oyster mushroom", 98: "shiitake", 99: "enoki mushroom", 100: "oyster mushroom",
    101: "white button mushroom", 102: "salad", 103: "other ingredients"
}

def convert_dataset(subset_size, dest_dir):
    """Downloads and converts HF FoodSeg103 semantic segmentation to YOLO format."""
    print("[Dataset] Loading FoodSeg103 dataset from Hugging Face...")
    ds = load_dataset("EduardoPacheco/FoodSeg103")
    
    splits = {
        "train": (ds["train"], "train"),
        "validation": (ds["validation"], "val")
    }
    
    for hf_split, (data, yolo_split) in splits.items():
        print(f"\n[Dataset] Converting '{hf_split}' split to '{yolo_split}'...")
        
        img_dir = os.path.join(dest_dir, "images", yolo_split)
        label_dir = os.path.join(dest_dir, "labels", yolo_split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        
        # Determine how many images to process
        total_items = len(data)
        limit = subset_size if subset_size > 0 else total_items
        if limit == total_items:
            print(f" -> Processing all {total_items} items.")
        else:
            # For validation subset, scale validation size proportionally (e.g. 50%)
            if yolo_split == "val" and subset_size > 0:
                limit = max(10, int(subset_size * 0.5))
            print(f" -> Processing subset of {limit} items (out of {total_items}).")
            
        for idx in range(limit):
            row = data[idx]
            img_id = row["id"]
            img = row["image"]
            label_mask = row["label"]
            
            # Save raw image
            img_path = os.path.join(img_dir, f"{img_id}.jpg")
            img.save(img_path)
            
            # Convert label mask to numpy array
            label_np = np.array(label_mask)
            h, w = label_np.shape
            
            # Find unique food classes present in this image
            unique_classes = np.unique(label_np)
            
            label_txt_path = os.path.join(label_dir, f"{img_id}.txt")
            with open(label_txt_path, "w") as label_file:
                for class_id in unique_classes:
                    # Skip background (0)
                    if class_id == 0:
                        continue
                    
                    # Create binary mask for this specific food category
                    bin_mask = (label_np == class_id).astype(np.uint8) * 255
                    
                    # Extract contours
                    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        # Simplify contour to reduce unnecessary collinear vertices
                        perimeter = cv2.arcLength(contour, True)
                        epsilon = 0.0015 * perimeter
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        if len(approx) < 3:
                            continue  # YOLO requires at least 3 points to form a polygon
                            
                        # Normalize points coordinates to [0, 1] range
                        normalized_points = []
                        for pt in approx:
                            px, py = pt[0]
                            normalized_points.append(px / w)
                            normalized_points.append(py / h)
                            
                        # Map 1-based class ID to 0-based YOLO index
                        yolo_class_id = class_id - 1
                        
                        # Write coordinates line
                        points_str = " ".join([f"{p:.6f}" for p in normalized_points])
                        label_file.write(f"{yolo_class_id} {points_str}\n")
                        
            if (idx + 1) % 50 == 0 or (idx + 1) == limit:
                print(f" -> Processed {idx + 1}/{limit} images.")

def create_yaml(dest_dir, abs_path):
    """Creates the dataset.yaml configuration file for YOLO training."""
    yaml_path = os.path.join(dest_dir, "dataset.yaml")
    
    # Map index 0-102 to class name
    names_dict = {i - 1: CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())}
    
    yaml_content = {
        "path": abs_path.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": names_dict
    }
    
    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_content, f, sort_keys=False)
        
    print(f"[Config] Saved training config file: '{yaml_path}'")
    return yaml_path

def main():
    parser = argparse.ArgumentParser(description="Convert FoodSeg103 and train YOLO segmentation model.")
    parser.add_argument(
        "--subset-size", 
        type=int, 
        default=100, 
        help="Number of images to convert for train split (use 0 or -1 for the FULL dataset)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3, 
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch", 
        type=int, 
        default=4, 
        help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=320, 
        help="Image size for training (default: 320)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        help="Device to train on (default: cpu)"
    )
    args = parser.parse_args()

    project_root = os.getcwd()
    dest_dir = os.path.join(project_root, "yolo_foodseg103")
    
    print("=" * 60)
    print("           YOLO FOOD SEGMENTATION TRAINING INITIATED          ")
    print("=" * 60)
    print(f"Project Root: '{project_root}'")
    print(f"Dataset Dir : '{dest_dir}'")
    print(f"Subset Size : {args.subset_size if args.subset_size > 0 else 'FULL DATASET'}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch Size  : {args.batch}")
    print(f"Image Size  : {args.imgsz}")
    print(f"Device      : {args.device}")
    print("=" * 60)

    # 1. Convert Dataset
    convert_dataset(args.subset_size, dest_dir)
    
    # 2. Generate YAML Config
    yaml_path = create_yaml(dest_dir, dest_dir)
    
    # 3. Train Model
    print(f"\n[Train] Initializing pre-trained YOLO11-seg base model (yolo11n-seg)...")
    model = YOLO("yolo11n-seg.pt")
    
    print(f"[Train] Starting fine-tuning...")
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/detect",
        name="foodseg103_finetuned"
    )
    
    print("\n" + "=" * 60)
    print("                 TRAINING RUN COMPLETE                  ")
    print("=" * 60)
    print(f"Finished training fine-tuned food segmentation model weights!")
    print(f"Weights are saved in: 'runs/detect/foodseg103_finetuned/weights/best.pt'")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
