# Project Progress Tracker

## Project Overview
A machine learning-based food waste estimator that compares before-and-after meal photos, estimates the percentage and nutritional value of food wasted, tracks long-term waste patterns, and provides recommendations for better portions and leftover strategies.

---

## Progress Log

### Phase 1: Environment & Initial Exploration
- [x] Read `description.txt` to understand requirements and specific instructions.
- [x] Explore the repository workspace structure and existing scripts (`basic_manipulation.py`, `basic_yolo.py`).
- [x] Initialize the internal memory tracking files (`progress.md` and `qa.md`).
- [x] Establish initial project design and implementation plan (under review).

### Phase 2: Core Algorithm Development
- [x] Implement food and container segmentation using YOLO11/YOLOv8-seg (fine-tuned on FoodSeg103).
- [x] Implement a comparison algorithm that calculates pixel/area differences between the "before" and "after" images.
- [x] Define baseline portion sizes and map food categories to nutrient data (calories, carbs, protein, fat) and carbon footprint values.
- [x] Build a database schema (SQLite) to store log entries of user meals, hunger levels, enjoyment, and waste reasons.

### Phase 3: Application Development
- [x] Build a robust back-end module to handle processing, estimation, and database storage.
- [x] Create an intuitive, visually stunning user interface using Streamlit to upload photos, input survey answers, and view analytics.
- [x] Design interactive charts showing waste patterns over time (categories, calories, CO2 impact).
- [x] Implement portion adjustment recommendations and recipe-based leftover strategies.


### Phase 4: Testing & Polish
- [x] Validate food detection and area estimation with test images (simulated test & custom arguments runner ready).
- [ ] Ensure accurate mapping of portion sizes and waste metrics.
- [ ] Refine the user interface with rich aesthetics, smooth animations, and clear data visualization.
- [ ] Review performance and complete the final documentation.
