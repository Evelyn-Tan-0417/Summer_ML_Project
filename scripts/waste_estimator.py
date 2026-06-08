"""
Smart Food Waste Estimator - Core CV & Calculation Pipeline
============================================================
This script loads:
1. yolov8n-seg.pt (to detect plate/bowl coordinates for normalization)
2. arunapb/yolo11l-food-segmentation (to detect and segment 103 detailed food ingredients)

It compares a 'before' and 'after' meal photo, normalizes the food pixel masks
against the plate dimensions, and outputs estimated waste weight, calories,
macronutrients, and carbon footprint (CO2 impact).
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# 1. NUTRITION & CO2 FOOTPRINT DATABASE (FoodSeg103 & COCO Mapping)
# ==============================================================================
# Profiles contain: (Typical Serving Size in grams, Calories per 100g, Protein g/100g, Carbs g/100g, Fat g/100g, CO2e kg/kg)
NUTRITION_DATABASE = {
    # Grains & Carbs
    "rice": (180, 130, 2.7, 28.0, 0.3, 1.5),
    "noodles": (200, 138, 4.5, 25.0, 2.1, 1.6),
    "pasta": (200, 158, 5.8, 30.9, 0.9, 1.8),
    "bread": (50, 265, 9.0, 49.0, 3.2, 1.2),
    "potato": (150, 87, 2.0, 20.0, 0.1, 0.5),
    "sweet potato": (150, 86, 1.6, 20.0, 0.1, 0.6),
    
    # Proteins / Meats
    "beef": (150, 250, 26.0, 0.0, 15.0, 27.0),
    "pork": (150, 242, 27.0, 0.0, 14.0, 12.1),
    "chicken": (150, 165, 31.0, 0.0, 3.6, 6.9),
    "fish": (150, 205, 22.0, 0.0, 12.0, 5.4),
    "egg": (100, 155, 13.0, 1.1, 11.0, 4.8),
    "tofu": (100, 76, 8.0, 1.9, 4.8, 1.0),
    "shrimp": (120, 99, 24.0, 0.2, 0.3, 12.0),
    
    # Vegetables & Salad
    "broccoli": (100, 34, 2.8, 7.0, 0.4, 0.4),
    "carrot": (100, 41, 0.9, 10.0, 0.2, 0.4),
    "tomato": (100, 18, 0.9, 3.9, 0.2, 0.8),
    "cabbage": (100, 25, 1.3, 6.0, 0.1, 0.4),
    "spinach": (100, 23, 2.9, 3.6, 0.4, 0.5),
    "onion": (50, 40, 1.1, 9.3, 0.1, 0.5),
    "lettuce": (100, 15, 1.4, 2.9, 0.2, 0.4),
    "asparagus": (100, 20, 2.2, 3.9, 0.1, 0.9),
    "green beans": (100, 31, 1.8, 7.0, 0.2, 0.5),
    
    # Fruits
    "apple": (150, 52, 0.3, 14.0, 0.2, 0.4),
    "banana": (120, 89, 1.1, 23.0, 0.3, 0.8),
    "avocado": (100, 160, 2.0, 8.5, 14.7, 2.5),
    "orange": (130, 47, 0.9, 12.0, 0.1, 0.5),
    
    # Dairy & Fats
    "cheese": (30, 402, 25.0, 1.3, 33.0, 8.5),
    "butter": (10, 717, 0.9, 0.1, 81.0, 9.0),
    
    # Default fallback profiles for unknown foods
    "food": (150, 150, 5.0, 20.0, 5.0, 2.5),
    "default": (150, 150, 5.0, 20.0, 5.0, 2.5)
}

class FoodWasteEstimator:
    def __init__(self):
        """Initialize and download YOLO models."""
        print("[System] Initializing Food Waste Estimator Models...")
        # yolov8n-seg is used to segment COCO objects (plates, bowls, cups, etc.)
        self.plate_model = YOLO("yolov8n-seg.pt")
        # arunapb/yolo11l-food-segmentation segments 103 detailed ingredients
        self.food_model = YOLO("arunapb/yolo11l-food-segmentation")
        print("[System] Models initialized successfully!")

    def segment_image(self, img_path):
        """
        Runs inference on an image and returns:
        - A dictionary of food items with their pixel mask area: {class_name: pixel_count}
        - The plate/bowl pixel mask area (if detected).
        - An annotated output image (BGR NumPy array).
        """
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")

        # Clone image for manual drawing
        annotated_img = img.copy()
        
        # 1. Run plate/container detection using yolov8n-seg
        plate_results = self.plate_model(img, verbose=False)
        plate_area = None
        plate_mask_combined = np.zeros(img.shape[:2], dtype=np.uint8)

        # COCO class IDs: 41 is 'cup', 45 is 'bowl', 61 is 'plate'
        CONTAINER_CLASSES = [41, 45, 61]
        
        for result in plate_results:
            if result.masks is not None:
                for idx, mask in enumerate(result.masks.data):
                    class_id = int(result.boxes.cls[idx])
                    if class_id in CONTAINER_CLASSES:
                        # Extract binary mask and resize to original image size
                        bin_mask = mask.cpu().numpy()
                        bin_mask_resized = cv2.resize(bin_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                        plate_mask_combined = cv2.bitwise_or(plate_mask_combined, bin_mask_resized.astype(np.uint8))
                        
                        # Draw semi-transparent cyan boundary for the container
                        contours, _ = cv2.findContours(plate_mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(annotated_img, contours, -1, (255, 255, 0), 2)
        
        # Count non-zero pixels inside plate mask
        plate_pixel_count = np.sum(plate_mask_combined > 0)
        if plate_pixel_count > 0:
            plate_area = float(plate_pixel_count)
            print(f" -> Reference container detected. Size: {plate_area:.0f} pixels.")
        else:
            print(" -> Warning: No plate/bowl reference container detected.")

        # 2. Run food ingredient detection using arunapb/yolo11l-food-segmentation
        food_results = self.food_model(img, verbose=False)
        food_masks = {}
        
        for result in food_results:
            if result.masks is not None:
                for idx, mask in enumerate(result.masks.data):
                    # Get class name
                    class_id = int(result.boxes.cls[idx])
                    class_name = result.names[class_id].lower()
                    
                    # Extract binary mask
                    bin_mask = mask.cpu().numpy()
                    bin_mask_resized = cv2.resize(bin_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                    
                    # Accumulate mask pixels if the food appears multiple times
                    if class_name in food_masks:
                        food_masks[class_name] = cv2.bitwise_or(food_masks[class_name], bin_mask_resized)
                    else:
                        food_masks[class_name] = bin_mask_resized
                        
                    # Draw a distinct color overlay for each food item
                    contours, _ = cv2.findContours(bin_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # Use a unique color based on class ID
                    np.random.seed(class_id)
                    color = tuple(int(c) for c in np.random.randint(50, 255, size=3))
                    cv2.drawContours(annotated_img, contours, -1, color, 2)
                    
                    # Add label text on the bounding box center
                    box = result.boxes.xyxy[idx].tolist()
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    cv2.putText(annotated_img, class_name, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Convert masks to pixel area sizes
        food_areas = {name: float(np.sum(mask > 0)) for name, mask in food_masks.items()}
        
        return food_areas, plate_area, annotated_img

    def estimate_waste(self, before_path, after_path):
        """
        Processes both 'before' and 'after' images, aligns them via normalized plate sizes,
        and calculates detailed waste analytics.
        """
        print(f"\n[CV] Analyzing BEFORE image: {os.path.basename(before_path)}")
        before_food_areas, before_plate, annotated_before = self.segment_image(before_path)
        
        print(f"[CV] Analyzing AFTER image: {os.path.basename(after_path)}")
        after_food_areas, after_plate, annotated_after = self.segment_image(after_path)
        
        # Determine normalization factor
        # If plate is found in both, normalize by plate size to account for distance/zoom shifts
        norm_factor = 1.0
        if before_plate is not None and after_plate is not None:
            norm_factor = before_plate / after_plate
            print(f"[CV] Camera zoom correction factor: {norm_factor:.3f} (Before plate: {before_plate:.0f}px, After plate: {after_plate:.0f}px)")
        else:
            print("[CV] No plates found in both images. Proceeding with raw pixel comparisons (unaligned).")

        # Core waste calculation
        waste_report = {}
        total_calories_wasted = 0.0
        total_protein_wasted = 0.0
        total_carbs_wasted = 0.0
        total_fat_wasted = 0.0
        total_co2_wasted = 0.0
        total_weight_wasted = 0.0

        print("\n" + "="*50)
        print("                 ESTIMATED FOOD WASTE REPORT             ")
        print("="*50)

        # Loop through foods present in the before image
        for food_name, before_area in before_food_areas.items():
            # Get the remaining area in the after image (corrected by the camera zoom factor)
            raw_after_area = after_food_areas.get(food_name, 0.0)
            corrected_after_area = raw_after_area * norm_factor
            
            # Avoid minor noise counting as leftovers (threshold = 1% of before area)
            if corrected_after_area < (before_area * 0.01):
                corrected_after_area = 0.0
                
            # Leftover percentage (capped at 100%)
            leftover_pct = min(100.0, (corrected_after_area / before_area) * 100.0)
            
            # Look up portion weight and nutrients
            db_profile = NUTRITION_DATABASE.get(food_name)
            if db_profile is None:
                # Try finding a partial match in keys (e.g. 'roasted chicken' -> 'chicken')
                db_profile = NUTRITION_DATABASE.get("default")
                for key, val in NUTRITION_DATABASE.items():
                    if key in food_name:
                        db_profile = val
                        break
            
            serving_size, kcal_factor, protein_factor, carbs_factor, fat_factor, co2_factor = db_profile
            
            # Calculations
            wasted_grams = serving_size * (leftover_pct / 100.0)
            wasted_kcal = wasted_grams * (kcal_factor / 100.0)
            wasted_protein = wasted_grams * (protein_factor / 100.0)
            wasted_carbs = wasted_grams * (carbs_factor / 100.0)
            wasted_fat = wasted_grams * (fat_factor / 100.0)
            wasted_co2 = (wasted_grams / 1000.0) * co2_factor  # CO2 footprint factor is per kg

            # Accumulate totals
            total_calories_wasted += wasted_kcal
            total_protein_wasted += wasted_protein
            total_carbs_wasted += wasted_carbs
            total_fat_wasted += wasted_fat
            total_co2_wasted += wasted_co2
            total_weight_wasted += wasted_grams

            waste_report[food_name] = {
                "leftover_percentage": leftover_pct,
                "wasted_weight_grams": wasted_grams,
                "calories_wasted": wasted_kcal,
                "protein_wasted_g": wasted_protein,
                "carbs_wasted_g": wasted_carbs,
                "fat_wasted_g": wasted_fat,
                "co2_wasted_kg": wasted_co2
            }

            print(f"- {food_name.upper()}:")
            print(f"  * Leftover Percentage : {leftover_pct:.1f}%")
            print(f"  * Mass Wasted        : {wasted_grams:.1f}g (of {serving_size}g baseline)")
            print(f"  * Calories Wasted    : {wasted_kcal:.1f} kcal")
            print(f"  * Carbon Footprint   : {wasted_co2:.3f} kg CO2e")

        print("="*50)
        print("                     SUMMARY TOTALS                      ")
        print("="*50)
        print(f"Total Mass Wasted       : {total_weight_wasted:.1f} g")
        print(f"Total Calories Wasted   : {total_calories_wasted:.1f} kcal")
        print(f"Total Protein Wasted    : {total_protein_wasted:.1f} g")
        print(f"Total Carbs Wasted      : {total_carbs_wasted:.1f} g")
        print(f"Total Fat Wasted        : {total_fat_wasted:.1f} g")
        print(f"Total CO2 Environmental : {total_co2_wasted:.3f} kg CO2e")
        print("="*50 + "\n")

        # Compile summaries
        summary = {
            "details": waste_report,
            "totals": {
                "wasted_weight_grams": total_weight_wasted,
                "calories_wasted": total_calories_wasted,
                "protein_wasted_g": total_protein_wasted,
                "carbs_wasted_g": total_carbs_wasted,
                "fat_wasted_g": total_fat_wasted,
                "co2_wasted_kg": total_co2_wasted
            }
        }
        
        return summary, annotated_before, annotated_after

# ==============================================================================
# TEST SIMULATION UTILITY (For demonstration and local validation)
# ==============================================================================
def create_simulated_after_image(before_path, output_path):
    """
    Creates a simulated 'after' image by loading a before image and masking out
    a rectangular section (representing eating part of the food).
    """
    img = cv2.imread(before_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {before_path} for simulation.")
        
    h, w, _ = img.shape
    simulated_after = img.copy()
    
    # Fill the left-to-middle section with a dark wood table pattern (simulating eaten food exposing table)
    # We will simply fill a circle representing eaten portion with dark grey/table color
    cv2.circle(simulated_after, (int(w * 0.45), int(h * 0.5)), int(min(w, h) * 0.35), (80, 80, 80), -1)
    
    cv2.imwrite(output_path, simulated_after)
    print(f"[Sim] Generated simulated 'after' meal photo at: '{output_path}'")
    return output_path

if __name__ == "__main__":
    # Setup test file directories
    test_before = "images/steak.jfif"
    results_dir = "results"
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    test_after = os.path.join(results_dir, "steak_simulated_after.jfif")
    
    # 1. Create a simulated leftover photo
    create_simulated_after_image(test_before, test_after)
    
    # 2. Run the pipeline
    estimator = FoodWasteEstimator()
    try:
        report, before_drawn, after_drawn = estimator.estimate_waste(test_before, test_after)
        
        # Save output drawing results
        before_save = os.path.join(results_dir, "steak_pipeline_before.jpg")
        after_save = os.path.join(results_dir, "steak_pipeline_after.jpg")
        
        cv2.imwrite(before_save, before_drawn)
        cv2.imwrite(after_save, after_drawn)
        print(f"[System] Pipeline runs successfully! Saved debug overlays to: \n  - '{before_save}'\n  - '{after_save}'")
    except Exception as e:
        print(f"Error executing pipeline: {e}")
