"""
Portion learning survey widgets to collect user feedback.
"""

import gradio as gr

def render_survey():
    """
    Renders survey inputs (hunger, fullness, enjoyment, reason for leftovers).
    
    Returns:
        tuple: (hunger_before, fullness_after, taste_enjoyment, reason_leftover)
    """
    with gr.Accordion("📋 Portion Learning Survey", open=True):
        hunger_before = gr.Slider(
            label="How hungry were you before eating? (1-10)", 
            minimum=1, 
            maximum=10, 
            step=1, 
            value=5
        )
        fullness_after = gr.Slider(
            label="How full were you after eating? (1-10)", 
            minimum=1, 
            maximum=10, 
            step=1, 
            value=8
        )
        taste_enjoyment = gr.Slider(
            label="How much did you like the meal? (1-10)", 
            minimum=1, 
            maximum=10, 
            step=1, 
            value=7
        )
        reason_leftover = gr.Dropdown(
            label="Why did you leave food?",
            choices=[
                "None", 
                "Too full", 
                "Didn't taste good", 
                "Portion too big", 
                "Too much carb", 
                "Too much vegetable", 
                "Texture issue", 
                "No time"
            ],
            value="None"
        )
    return hunger_before, fullness_after, taste_enjoyment, reason_leftover
