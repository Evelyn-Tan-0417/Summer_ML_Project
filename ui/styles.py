"""
Custom modern CSS styling for the Gradio application.
Separated to keep styling independent and maintainable.
"""

CSS = """
/* Purple Theme Custom CSS Variables */
:root {
    --body-background-fill: #f5f3ff !important; /* Soft lavender page background */
    --block-background-fill: #ffffff !important; /* Clean white block card background */
    --block-border-color: #ddd6fe !important; /* Soft purple border */
    --block-border-width: 1px !important;
    --block-title-text-color: #6d28d9 !important; /* Deep violet block headers */
    --input-background-fill: #faf5ff !important; /* Light purple inputs */
    --input-border-color: #e9d5ff !important;
    --slider-color: #7c3aed !important;
    
    /* Button primary styling */
    --button-primary-background-fill: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    --button-primary-background-fill-hover: linear-gradient(135deg, #6d28d9, #5b21b6) !important;
    --button-primary-text-color: #ffffff !important;
}

.dark {
    --body-background-fill: #0c0714 !important; /* Very dark rich purple page background */
    --block-background-fill: #171123 !important; /* Dark purple block card background */
    --block-border-color: #3b2c54 !important;
    --block-title-text-color: #c084fc !important;
    --input-background-fill: #221b30 !important;
    --input-border-color: #3b2c54 !important;
    --slider-color: #a78bfa !important;
    
    --button-primary-background-fill: linear-gradient(135deg, #8b5cf6, #7c3aed) !important;
    --button-primary-background-fill-hover: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    --button-primary-text-color: #ffffff !important;
}

body {
    background-color: var(--body-background-fill) !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    padding: 20px !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}
.header-title {
    background: linear-gradient(90deg, #7c3aed, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 4px;
    text-align: center;
    letter-spacing: -0.5px;
}
.header-subtitle {
    font-size: 16px;
    color: #6b7280;
    text-align: center;
    margin-bottom: 32px;
}
.gradio-button {
    background: linear-gradient(135deg, #7c3aed, #5b21b6) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.2) !important;
    padding: 14px 28px !important;
    font-size: 16px !important;
    transition: all 0.2s ease-in-out !important;
    border: none !important;
    cursor: pointer !important;
    display: block !important;
    width: 100% !important;
}
.gradio-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(124, 58, 237, 0.3) !important;
}
.gradio-button:active {
    transform: translateY(0) !important;
}
"""
