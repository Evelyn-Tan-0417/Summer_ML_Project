"""
OpenCV Basic Image Manipulation Tutorial
=========================================
This script teaches you the fundamental building blocks of OpenCV in Python:
1. Loading / reading an image from disk.
2. Understanding image structure (dimensions, data types, BGR channel order).
3. Displaying images in a window.
4. Converting between color spaces (BGR, RGB, Grayscale).
5. Flipping images (horizontally, vertically, both).
6. Drawing annotations (lines, rectangles, circles, and text).
7. Saving the modified images back to disk.

Make sure you run this script using:
    pixi run python scripts/basic_manipulation.py
"""

import os
import cv2
import numpy as np

def run_tutorial():
    print("=" * 60)
    print("         OPENCV BASIC IMAGE MANIPULATION TUTORIAL         ")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1: Define Path and Load the Image
    # ---------------------------------------------------------
    # We will use one of the food images in the 'images' directory
    image_dir = "images"
    image_name = "steak.jfif"
    image_path = os.path.join(image_dir, image_name)

    print(f"\n[Step 1] Loading image from: '{image_path}'...")
    
    # cv2.imread() reads the image from the specified file path.
    # By default, cv2.imread() loads color images in BGR (Blue, Green, Red) format.
    # Even if the image is grayscale on disk, it will still have 3 channels unless specified.
    img = cv2.imread(image_path)

    # Always verify if the image loaded successfully!
    if img is None:
        print(f"ERROR: Could not load the image at '{image_path}'.")
        print("Please check if the file path is correct and the file exists.")
        return
    else:
        print("Successfully loaded the image!")

    # ---------------------------------------------------------
    # STEP 2: Understand Image Properties (Metadata)
    # ---------------------------------------------------------
    print("\n[Step 2] Understanding Image Properties...")
    # In OpenCV, images are loaded as standard NumPy n-dimensional arrays (ndarrays).
    # img.shape returns a tuple representing (Height, Width, Channels)
    height, width, channels = img.shape
    print(f" - Image Type: {type(img)}")
    print(f" - Dimensions (Height x Width): {height} x {width} pixels")
    print(f" - Number of Channels: {channels} (BGR)")
    # img.dtype tells us the data type of the pixels. Usually uint8 (unsigned 8-bit integer, 0-255).
    print(f" - Data Type of Pixels: {img.dtype}")

    # Let's show the original image using cv2.imshow()
    # cv2.imshow() takes (window_name, image_numpy_array)
    print("\n--> Displaying original image in a window. Press ANY KEY in the window to continue...")
    cv2.imshow("Original BGR Image", img)
    # cv2.waitKey(0) pauses execution until any keyboard key is pressed inside the window.
    # Passing 0 (or no value) means it waits indefinitely.
    cv2.waitKey(0)
    cv2.destroyAllWindows() # Close the window to keep the desktop clean

    # ---------------------------------------------------------
    # STEP 3: Color Space Conversions
    # ---------------------------------------------------------
    print("\n[Step 3] Performing Color Space Conversions...")
    # OpenCV reads images as BGR by default, but most Python libraries (like Matplotlib) 
    # and AI models expect RGB (Red, Green, Blue). Grayscale is also commonly used for 
    # image processing (like edge detection) because it reduces computational complexity.

    # 3.1 BGR to Grayscale (single channel, intensity from 0 to 255)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f" - Grayscale Image Shape: {img_gray.shape} (Note: no channel dimension, 1 channel)")

    # 3.2 BGR to RGB (needed for matplotlib, pygame, or some deep learning models)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(" - Converted BGR image to RGB format.")

    # Show Grayscale image
    print("\n--> Displaying Grayscale image. Press ANY KEY in the window to continue...")
    cv2.imshow("Grayscale Image", img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ---------------------------------------------------------
    # STEP 4: Image Flipping
    # ---------------------------------------------------------
    print("\n[Step 4] Flipping the Image...")
    # cv2.flip(src, flipCode):
    #   flipCode = 1  --> Horizontal flip (left-to-right)
    #   flipCode = 0  --> Vertical flip (upside-down)
    #   flipCode = -1 --> Both horizontal and vertical flip
    
    flip_horiz = cv2.flip(img, 1)
    flip_vert = cv2.flip(img, 0)
    flip_both = cv2.flip(img, -1)

    print(" - Flipped image horizontally, vertically, and both.")
    
    # Show the horizontal flip as an example
    print("\n--> Displaying Horizontally Flipped image. Press ANY KEY in the window to continue...")
    cv2.imshow("Horizontal Flip", flip_horiz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ---------------------------------------------------------
    # STEP 5: Adding Annotations (Drawing)
    # ---------------------------------------------------------
    print("\n[Step 5] Adding Annotations (Drawing)...")
    # IMPORTANT: OpenCV drawing functions modify the input image IN-PLACE!
    # If we want to keep the original image intact, we must create a copy first.
    annotated_img = img.copy()

    # Let's define some colors in BGR format: BGR = (Blue, Green, Red)
    # Values range from 0 to 255.
    COLOR_RED = (0, 0, 255)       # B=0,   G=0,   R=255
    COLOR_GREEN = (0, 255, 0)     # B=0,   G=255, R=0
    COLOR_BLUE = (255, 0, 0)      # B=255, G=0,   R=0
    COLOR_YELLOW = (0, 255, 255)  # B=0,   G=255, R=255
    COLOR_WHITE = (255, 255, 255) # B=255, G=255, R=255

    # 5.1 Draw a Line
    # cv2.line(img, start_point(x, y), end_point(x, y), color, thickness_pixels)
    start_pt = (10, 10)
    end_pt = (width - 10, 10)
    cv2.line(annotated_img, start_pt, end_pt, COLOR_BLUE, thickness=4)
    print(" - Drew a BLUE horizontal line at the top.")

    # 5.2 Draw a Rectangle (often used for YOLO bounding boxes!)
    # cv2.rectangle(img, top_left_corner(x, y), bottom_right_corner(x, y), color, thickness_pixels)
    # We will draw a box near the center of the image.
    box_top_left = (int(width * 0.2), int(height * 0.2))
    box_bottom_right = (int(width * 0.8), int(height * 0.8))
    cv2.rectangle(annotated_img, box_top_left, box_bottom_right, COLOR_GREEN, thickness=3)
    print(" - Drew a GREEN rectangle (bounding box).")

    # 5.3 Draw a Circle
    # cv2.circle(img, center_point(x, y), radius_pixels, color, thickness_pixels)
    # Note: If thickness is -1, the circle is FILLED.
    center = (int(width / 2), int(height / 2))
    radius = 30
    cv2.circle(annotated_img, center, radius, COLOR_RED, thickness=-1)  # -1 makes it a solid filled circle
    print(" - Drew a filled RED circle in the exact center of the image.")

    # 5.4 Put Text
    # cv2.putText(img, text, bottom_left_origin(x, y), fontFace, fontScale, color, thickness, lineType)
    # Font scale is a multiplier, thickness is in pixels. cv2.LINE_AA is anti-aliased (highly recommended).
    text = "Steak Dinner!"
    text_position = (int(width * 0.25), int(height * 0.75))
    cv2.putText(
        annotated_img, 
        text, 
        text_position, 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=1.0, 
        color=COLOR_YELLOW, 
        thickness=2, 
        lineType=cv2.LINE_AA
    )
    print(" - Placed YELLOW text 'Steak Dinner!' on the image.")

    # Show the final annotated image
    print("\n--> Displaying annotated image. Press ANY KEY in the window to continue...")
    cv2.imshow("Annotated Image", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ---------------------------------------------------------
    # STEP 6: Saving Images to Disk
    # ---------------------------------------------------------
    print("\n[Step 6] Saving Images to Disk...")
    # Create an output directory to store the results
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f" - Created output directory: '{output_dir}/'")

    # Save images using cv2.imwrite(filename, image_numpy_array)
    # OpenCV handles the file format based on the extension (.jpg, .png, etc.)
    gray_save_path = os.path.join(output_dir, "steak_grayscale.jpg")
    annotated_save_path = os.path.join(output_dir, "steak_annotated.jpg")
    horiz_flip_save_path = os.path.join(output_dir, "steak_flipped.jpg")

    cv2.imwrite(gray_save_path, img_gray)
    cv2.imwrite(annotated_save_path, annotated_img)
    cv2.imwrite(horiz_flip_save_path, flip_horiz)

    print(f" - Saved grayscale image to: '{gray_save_path}'")
    print(f" - Saved annotated image to: '{annotated_save_path}'")
    print(f" - Saved horizontally flipped image to: '{horiz_flip_save_path}'")

    print("\n" + "=" * 60)
    print(" TUTORIAL COMPLETED! Check out the saved files in 'results/' ")
    print("=" * 60)

if __name__ == "__main__":
    run_tutorial()
