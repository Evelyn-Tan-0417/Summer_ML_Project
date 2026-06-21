"""
Custom modern CSS styling for the Gradio application.
Separated to keep styling independent and maintainable.
"""

CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    padding: 20px !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}
.header-title {
    background: linear-gradient(90deg, #3182ce, #319795, #2f855a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 4px;
    text-align: center;
    letter-spacing: -0.5px;
}
.header-subtitle {
    font-size: 16px;
    color: #4a5568;
    text-align: center;
    margin-bottom: 32px;
}
.gradio-button {
    background: linear-gradient(135deg, #3182ce, #2b6cb0) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(49, 130, 206, 0.2) !important;
    padding: 12px 24px !important;
    font-size: 15px !important;
    transition: all 0.2s ease-in-out !important;
    border: none !important;
    cursor: pointer !important;
}
.gradio-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(49, 130, 206, 0.3) !important;
}
.gradio-button:active {
    transform: translateY(0) !important;
}
"""
