"""
Smart Food Waste Estimator - Gradio Web Application
===================================================
A beautiful, premium web dashboard that allows users to upload before-and-after
eating photos, submit meal satisfaction surveys, log results to a relational SQLite database,
and view detailed portions recommendations and environmental reports.

Refactored to be modular so multiple developers can edit widgets independently without conflict.

To run:
  pixi run python gradio_app.py
"""

import os
import sys
import gradio as gr

# Ensure scripts directory is in path for compatibility
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

# Import independent modular UI components and callback handlers
from ui import (
    CSS,
    render_header,
    render_image_upload,
    render_survey,
    render_action_button,
    render_results_preview,
    analyze_waste,
)

# Construct UI using gr.Blocks and modular widgets
with gr.Blocks(title="Smart Food Waste Estimator") as demo:
    # 1. Header widget
    render_header()
    
    with gr.Row():
        # 2. Left Panel: Inputs & Surveys
        with gr.Column(scale=1):
            # Modular image uploads
            before_input, after_input = render_image_upload()
            
            # Modular survey inputs
            hunger_before, fullness_after, taste_enjoyment, reason_leftover = render_survey()
            
            # Modular action trigger button
            run_btn = render_action_button()
            
        # 3. Right Panel: Results & Analytics
        with gr.Column(scale=2):
            # Modular results and report dashboards
            before_overlay, after_overlay, output_html = render_results_preview()
            
    # Connect event triggers (bound to independent modules)
    run_btn.click(
        fn=analyze_waste,
        inputs=[before_input, after_input, hunger_before, fullness_after, taste_enjoyment, reason_leftover],
        outputs=[before_overlay, after_overlay, output_html]
    )

if __name__ == "__main__":
    # Launch with custom CSS
    try:
        demo.launch(share=False, css=CSS)
    except TypeError:
        # Fallback if launch doesn't support css parameter in very old versions
        demo.css = CSS
        demo.launch(share=False)
