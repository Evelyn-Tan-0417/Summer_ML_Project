"""
Smart Food Waste Estimator - Show Segmentation Overlays
======================================================
This script runs the food and plate segmentation models on the before and after photos
and displays the annotated results side-by-side.

Usage:
  pixi run python scripts/show_segmentation.py --before <path_to_before> --after <path_to_after>
"""

import os
import cv2
import argparse
import numpy as np
from waste_estimator import FoodWasteEstimator

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

def main():
    parser = argparse.ArgumentParser(description="Show AI Segmentation boundaries on before/after photos.")
    parser.add_argument(
        "--before", 
        type=str, 
        default="images/steak.jfif",
        help="Path to the before-eating photo"
    )
    parser.add_argument(
        "--after", 
        type=str, 
        default="", 
        help="Path to the after-eating photo"
    )
    args = parser.parse_args()

    before_path = resolve_path(args.before)
    after_path = resolve_path(args.after)

    if not os.path.exists(before_path):
        print(f"Error: Before-eating photo not found at: '{before_path}'")
        return

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # If no after image is provided, simulate one
    if not after_path:
        print("[Show] No after-eating photo provided. Simulating one...")
        after_path = os.path.join(results_dir, "steak_simulated_after.jfif")
        img = cv2.imread(before_path)
        if img is None:
            print(f"Error: Could not read image at '{before_path}'")
            return
        h, w, _ = img.shape
        sim_img = img.copy()
        cv2.circle(sim_img, (int(w * 0.45), int(h * 0.5)), int(min(w, h) * 0.35), (80, 80, 80), -1)
        cv2.imwrite(after_path, sim_img)

    if not os.path.exists(after_path):
        print(f"Error: After-eating photo not found at: '{after_path}'")
        return

    print(f"\n[Show] Initializing models and running segmentation...")
    try:
        estimator = FoodWasteEstimator()
        
        # Segment both images
        print(f" -> Processing Before image: '{before_path}'")
        _, _, before_annotated = estimator.segment_image(before_path)
        
        print(f" -> Processing After image: '{after_path}'")
        _, _, after_annotated = estimator.segment_image(after_path)
        
        # Resize images to match heights for side-by-side display
        h1, w1, _ = before_annotated.shape
        h2, w2, _ = after_annotated.shape
        target_h = 600  # Target display height
        
        before_resized = cv2.resize(before_annotated, (int(w1 * target_h / h1), target_h))
        after_resized = cv2.resize(after_annotated, (int(w2 * target_h / h2), target_h))
        
        # Add labels to the images
        cv2.putText(before_resized, "BEFORE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(after_resized, "AFTER", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        
        # Combine side-by-side
        combined_img = np.hstack((before_resized, after_resized))
        
        # Save output
        save_path = os.path.join(results_dir, "segmentation_comparison.jpg")
        cv2.imwrite(save_path, combined_img)
        print(f"[Show] Comparison image saved to: '{save_path}'")
        
        # Display the window
        window_name = "Food & Plate Segmentation (Before vs After)"
        print("\n[Show] Displaying segmentation. Press any key in the GUI window to close it...")
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(window_name, combined_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("[Show] Window closed.")
        except cv2.error as cv_err:
            print(f"[Show] Note: Could not open display window ({cv_err}). Comparison image was successfully saved to '{save_path}'.")
        
    except Exception as e:
        print(f"\n[Show] Failed to show segmentation: {e}")

if __name__ == "__main__":
    main()
