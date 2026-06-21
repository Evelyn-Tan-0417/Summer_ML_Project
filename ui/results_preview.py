"""
Results and dashboard preview widgets.
"""

import gradio as gr

def render_results_preview():
    """
    Renders the results section, including BGR/RGB before-and-after segmentation masks
    and the final HTML analytics dashboard.
    
    Returns:
        tuple: (before_overlay_img, after_overlay_img, dashboard_html)
    """
    gr.Markdown("### 📊 AI Overlay & Segmentation Output")
    
    with gr.Row():
        before_overlay = gr.Image(label="Before Photo Overlay", interactive=False)
        after_overlay = gr.Image(label="After Photo Overlay", interactive=False)
    
    gr.Markdown("### 📈 Nutritional & Environmental Impact Dashboard")
    output_html = gr.HTML(
        value="""
        <div style="text-align:center; color:#718096; padding:60px 40px; border:2px dashed #cbd5e0; border-radius:12px; font-weight:600; background:#f7fafc;">
            📸 Upload your meal photos, submit the survey, and click 'Analyze Meal Waste' to view the AI analysis.
        </div>
        """
    )
    return before_overlay, after_overlay, output_html
