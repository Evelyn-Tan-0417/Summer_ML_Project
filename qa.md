# Project Q&A Log

This file stores all questions asked during development and their corresponding short answers.

| Date | Question | Short Answer |
| :--- | :--- | :--- |
| 2026-06-01 | How is the proposed YOLOv8-seg better than FoodSeg103? | YOLOv8-seg is a model architecture, while FoodSeg103 is a dataset. They are complementary: we can fine-tune a YOLOv8-seg model on the FoodSeg103 dataset to achieve high-accuracy food segmentation across 103 classes. |
| 2026-06-01 | SQLite vs. JSON for database? | JSON is easy to view but slow and lacks indexing. SQLite is a lightweight, zero-config relational database supporting SQL queries for patterns, making it highly recommended for long-term analytics. |
| 2026-06-01 | Gradio vs. Streamlit for UI? | Gradio is best for side-by-side ML inputs/outputs. Streamlit is superior for multi-page layouts and interactive data analytics. Streamlit is recommended for combining meal logging with dashboard charts. |
| 2026-06-01 | For SQLite, should we use SQL or NoSQL? | SQLite is natively a relational (SQL) database. We will use standard SQL tables and queries to store our structured logs, enabling easy aggregation and trend calculations. |
| 2026-06-01 | Can we leverage fine-tuned YOLO11 on FoodSeg103? | Yes. We can load the pre-trained `arunapb/yolo11l-food-segmentation` model directly from Hugging Face via Ultralytics. It segment 103 food classes out of the box and is ideal for this project. |
| 2026-06-01 | Are you going to consider the before and after picture of the food plate? | Yes. We will process both photos. To handle camera angle and distance shifts between shots, we will segment the plate/bowl as a reference, then normalize or align the food areas relative to it. |

