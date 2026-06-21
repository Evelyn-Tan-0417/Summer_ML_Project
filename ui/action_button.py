"""
Action trigger button component.
"""

import gradio as gr

def render_action_button():
    """
    Renders the primary execution button for running the AI analysis pipeline.
    
    Returns:
        gr.Button: The action button.
    """
    return gr.Button("Analyze Meal Waste", elem_classes="gradio-button")
