"""
Smart Food Waste Estimator - Custom Image Pipeline Test Runner
=============================================================
This script allows you to easily test the computer vision pipeline with your own
before-and-after meal photos by passing them as command-line arguments.

Usage:
  pixi run python scripts/run_custom_test.py --before <path_to_before> --after <path_to_after>
"""

import os
import cv2
import argparse
from waste_estimator import FoodWasteEstimator

def main():
    parser = argparse.ArgumentParser(description="Test the AI Food Waste Estimation CV Pipeline.")
    parser.add_argument(
        "--before", 
        type=str, 
        default="images/steak.jfif",
        help="Path to the before-eating photo (default: images/steak.jfif)"
    )
    parser.add_argument(
        "--after", 
        type=str, 
        default="", 
        help="Path to the after-eating photo (if blank, a simulated after photo will be generated)"
    )
    args = parser.parse_args()

    def resolve_path(path):
        if not path or os.path.exists(path):
            return path
        if path.lower().endswith('.jpg'):
            alt = path[:-4] + '.jpeg'
            if os.path.exists(alt):
                return alt
        elif path.lower().endswith('.jpeg'):
            alt = path[:-5] + '.jpg'
            if os.path.exists(alt):
                return alt
        return path

    before_path = resolve_path(args.before)
    after_path = resolve_path(args.after)

    # Validation
    if not os.path.exists(before_path):
        print(f"Error: Before-eating photo not found at: '{before_path}'")
        return

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # If no after image is provided, simulate one
    if not after_path:
        print("[Test] No after-eating photo provided. Simulating an after-eating photo from the before image...")
        after_path = os.path.join(results_dir, "steak_simulated_after.jfif")
        
        # Load before and mask out a circle
        img = cv2.imread(before_path)
        if img is None:
            print(f"Error: Could not read image at '{before_path}'")
            return
        h, w, _ = img.shape
        sim_img = img.copy()
        cv2.circle(sim_img, (int(w * 0.45), int(h * 0.5)), int(min(w, h) * 0.35), (80, 80, 80), -1)
        cv2.imwrite(after_path, sim_img)
        print(f"[Test] Simulated photo saved to: '{after_path}'")

    if not os.path.exists(after_path):
        print(f"Error: After-eating photo not found at: '{after_path}'")
        return

    # Run pipeline
    print(f"\n[Test] Running pipeline...")
    print(f" -> Before: '{before_path}'")
    print(f" -> After : '{after_path}'")
    
    try:
        estimator = FoodWasteEstimator()
        report, before_drawn, after_drawn = estimator.estimate_waste(before_path, after_path)
        
        # Save output overlays
        before_fn = os.path.basename(before_path).split('.')[0]
        after_fn = os.path.basename(after_path).split('.')[0]
        
        before_save = os.path.join(results_dir, f"test_{before_fn}_overlay.jpg")
        after_save = os.path.join(results_dir, f"test_{after_fn}_overlay.jpg")
        
        cv2.imwrite(before_save, before_drawn)
        cv2.imwrite(after_save, after_drawn)
        
        print("\n" + "=" * 60)
        print("                  PIPELINE RUN COMPLETED                 ")
        print("=" * 60)
        print(f"Saved annotated preview images with AI boundaries in '{results_dir}/':")
        print(f"  1. Before Photo Overlay: '{before_save}'")
        print(f"  2. After Photo Overlay : '{after_save}'")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n[Test] Pipeline failed with error: {e}")
        print("💡 Note: Ensure your environment has fully finished downloading PyTorch/Ultralytics.")

if __name__ == "__main__":
    main()
