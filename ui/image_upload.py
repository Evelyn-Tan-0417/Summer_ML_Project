"""
Image upload widgets for before and after eating photos.
"""

import gradio as gr

def render_image_upload():
    """
    Renders before and after image upload inputs.
    
    Returns:
        tuple: (before_image_input, after_image_input)
    """
    gr.Markdown("### 📸 Upload Meal Photos")
    before_input = gr.Image(label="Before-Eating Photo (Required)", type="filepath")
    after_input = gr.Image(label="After-Eating Photo (Optional)", type="filepath")
    gr.Markdown("*Note: If you do not upload an after-eating photo, the AI will automatically generate a simulated leftovers image.*")
    return before_input, after_input
