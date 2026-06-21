"""
UI components package for Smart Food Waste Estimator.
Each widget module is independent, making it easy to rearrange layout or add new widgets.
"""

from .styles import CSS
from .header import render_header
from .image_upload import render_image_upload
from .survey import render_survey
from .action_button import render_action_button
from .results_preview import render_results_preview
from .callbacks import analyze_waste

__all__ = [
    "CSS",
    "render_header",
    "render_image_upload",
    "render_survey",
    "render_action_button",
    "render_results_preview",
    "analyze_waste",
]
