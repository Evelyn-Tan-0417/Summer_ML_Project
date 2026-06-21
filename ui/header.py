"""
Header component widget for the food waste estimator web app.
"""

import gradio as gr

def render_header():
    """
    Renders the HTML title and subtitle for the application header.
    """
    return gr.HTML(
        """
        <div class="header-title">🍲 Smart Food Waste Estimator</div>
        <div class="header-subtitle">AI-powered portion tracking, nutritional waste analytics, and CO2 environmental impact reporting</div>
        """
    )
