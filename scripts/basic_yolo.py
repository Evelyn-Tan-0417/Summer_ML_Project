"""
OpenCV & YOLOv8 Object Detection Tutorial
=========================================
This script teaches you how to perform object detection using a pre-trained YOLOv8 model.
It covers:
1. Loading a pre-trained YOLOv8 model (yolov8n.pt - nano version).
2. Running inference on an image from the 'images' folder.
3. Accessing the underlying prediction results (boxes, confidence scores, classes).
4. Option A (Easy): Using Ultralytics' built-in .plot() function to draw boxes.
5. Option B (Advanced & OpenCV-focused): Manually iterating through the predictions, 
   extracting the coordinates, and drawing the boxes and labels using cv2.rectangle() 
   and cv2.putText() ourselves.
6. Saving the detection results to the 'results' folder.

Before running, ensure 'ultralytics' is installed in your pixi environment.
Run this script using:
    pixi run python scripts/basic_yolo.py
"""

import os
import cv2
from ultralytics import YOLO

def run_yolo_tutorial():
    print("=" * 60)
    print("           YOLOv8 OBJECT DETECTION TUTORIAL               ")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1: Load a Pre-Trained YOLOv8 Model
    # ---------------------------------------------------------
    print("\n[Step 1] Loading pre-trained YOLOv8 model...")
    # We will use 'yolov8n.pt' (YOLOv8 Nano). It is extremely lightweight,
    # runs very fast on standard CPUs, and is pre-trained on the COCO dataset 
    # (which includes common objects like food, plates, forks, knives, people, cars, etc.).
    # If the model file is not already downloaded, the library will download it automatically.
    model = YOLO("yolov8n.pt")
    print("YOLOv8 Nano model loaded successfully!")

    # ---------------------------------------------------------
    # STEP 2: Load the Target Image
    # ---------------------------------------------------------
    image_dir = "images"
    image_name = "steak.jfif"  # Let's try detecting on steak.jfif or lasagna.jfif
    image_path = os.path.join(image_dir, image_name)

    print(f"\n[Step 2] Loading target image from '{image_path}'...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not load the image at '{image_path}'.")
        return
    print(f"Loaded image. Size: {img.shape[1]}x{img.shape[0]}")

    # ---------------------------------------------------------
    # STEP 3: Run YOLOv8 Inference
    # ---------------------------------------------------------
    print("\n[Step 3] Running YOLOv8 object detection...")
    # We simply pass the image path or the loaded BGR image array to the model.
    # The model returns a list of 'Results' objects (one per input image).
    results = model(image_path)
    
    # Grab the first result (since we only passed one image)
    result = results[0]
    
    # Let's see what we found!
    num_detections = len(result.boxes)
    print(f"Detection completed. Found {num_detections} object(s) in the image.")

    # ---------------------------------------------------------
    # STEP 4: Option A - Automatic Visualization (The Easy Way)
    # ---------------------------------------------------------
    print("\n[Step 4] Option A: Automatic Visualization using Ultralytics...")
    # result.plot() returns a standard BGR numpy array with all boxes and labels pre-drawn.
    annotated_img_auto = result.plot()

    # Show the automatically annotated image
    print("--> Displaying automatic detection. Press ANY KEY in the window to continue...")
    cv2.imshow("YOLOv8 Detections (Auto Plot)", annotated_img_auto)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the automatic result
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    auto_save_path = os.path.join(output_dir, "yolo_detected_auto.jpg")
    cv2.imwrite(auto_save_path, annotated_img_auto)
    print(f" - Saved automatic detection image to: '{auto_save_path}'")

    # ---------------------------------------------------------
    # STEP 5: Option B - Manual Annotation using OpenCV (The Learning Way)
    # ---------------------------------------------------------
    print("\n[Step 5] Option B: Manual Annotation using OpenCV...")
    # This is where we apply what we learned in basic_manipulation.py!
    # We will copy the original image and draw our own custom boxes and labels.
    annotated_img_manual = img.copy()

    # YOLOv8 predictions are stored in result.boxes
    # Each box contains:
    #   - box.xyxy: Coordinates of the bounding box [xmin, ymin, xmax, ymax]
    #   - box.conf: Confidence score (probability, e.g., 0.85)
    #   - box.cls:  Class ID (index of the detected class)
    #
    # We can look up the text label using model.names[class_id]

    print("Iterating through detected objects:")
    for idx, box in enumerate(result.boxes):
        # Extract coordinates (xyxy is a list/tensor. We take the first element and convert to float/int)
        coords = box.xyxy[0].tolist()
        xmin, ymin, xmax, ymax = map(int, coords) # Convert coordinates to integers for OpenCV

        # Extract confidence score and convert to percentage
        conf = float(box.conf[0])
        conf_percentage = conf * 100

        # Extract class ID and name
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        print(f" {idx + 1}. Object: '{class_name}' | Confidence: {conf:.2f} ({conf_percentage:.1f}%) | Box: [{xmin}, {ymin}, {xmax}, {ymax}]")

        # Define custom styling for drawing
        COLOR_BOX = (0, 255, 0)      # Green bounding box (BGR)
        COLOR_TEXT = (255, 255, 255)  # White text (BGR)
        COLOR_BG = (0, 165, 255)     # Orange background for label (BGR)

        # 5.1 Draw the Bounding Box
        cv2.rectangle(annotated_img_manual, (xmin, ymin), (xmax, ymax), COLOR_BOX, thickness=2)

        # 5.2 Draw a Label Background banner (filled rectangle)
        # Create a nice label string: e.g. "dining table: 87%" or "steak: 92%" (Note: COCO dataset names steak/food as "hot dog", "pizza", "dining table", etc.!)
        label_text = f"{class_name}: {conf_percentage:.0f}%"
        
        # Get the size of the text box to draw a background rectangle behind it
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1
        )
        
        # Make sure the text background doesn't go off the screen
        label_ymin = max(ymin, text_height + 10)
        
        # Draw orange filled rectangle for text background
        cv2.rectangle(
            annotated_img_manual, 
            (xmin, label_ymin - text_height - 6), 
            (xmin + text_width + 10, label_ymin + baseline - 4), 
            COLOR_BG, 
            thickness=-1
        )

        # 5.3 Put the label text inside the orange banner
        cv2.putText(
            annotated_img_manual,
            label_text,
            (xmin + 5, label_ymin - 4),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=COLOR_TEXT,
            thickness=1,
            lineType=cv2.LINE_AA
        )

    # Show the custom manual annotated image
    print("\n--> Displaying custom manual detection. Press ANY KEY in the window to continue...")
    cv2.imshow("YOLOv8 Detections (Manual OpenCV)", annotated_img_manual)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the manual result
    manual_save_path = os.path.join(output_dir, "yolo_detected_manual.jpg")
    cv2.imwrite(manual_save_path, annotated_img_manual)
    print(f" - Saved manual detection image to: '{manual_save_path}'")

    print("\n" + "=" * 60)
    print(" YOLOv8 DETECTION TUTORIAL COMPLETED! Results saved in 'results/' ")
    print("=" * 60)

if __name__ == "__main__":
    run_yolo_tutorial()
