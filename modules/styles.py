"""
Module for handling application styling
"""

def load_css():
    return """
    <style>
    /* Main Theme */
    .main-bg {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        min-height: 100vh;
    }
    
    /* Glass Card Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(45deg, #4CAF50, #81C784);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Modern Form Elements */
    .modern-input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
        padding: 0.75rem;
    }
    
    /* Buttons */
    .modern-button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    /* Metrics Display */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Progress Bars */
    .progress-bar {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        overflow: hidden;
        height: 6px;
    }
    
    .progress-value {
        background: linear-gradient(45deg, #4CAF50, #81C784);
        height: 100%;
        transition: width 0.3s ease;
    }
    
    /* Weather Cards */
    .weather-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        color: white;
    }
    
    /* Alerts and Notifications */
    .alert {
        background: rgba(255, 255, 255, 0.1);
        border-left: 4px solid;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
    }
    
    .alert-success { border-color: #4CAF50; }
    .alert-warning { border-color: #FFC107; }
    .alert-error { border-color: #F44336; }
    
    /* Responsive Grid */
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
    }
    
    /* Charts and Visualizations */
    .chart-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Responsive Enhancements */
    @media (max-width: 768px) {
        .glass-card, .weather-card, .metric-card, .knowledge-card {
            padding: 0.75rem !important;
            margin: 0.5rem 0 !important;
            border-radius: 12px !important;
        }
        .grid-container {
            grid-template-columns: 1fr !important;
            gap: 0.5rem !important;
        }
        h1, h2, h3, h4 {
            font-size: 1.2em !important;
        }
        table, th, td {
            font-size: 0.95em !important;
        }
        .modern-form, .modern-input {
            padding: 0.5rem !important;
        }
    }
    @media (max-width: 480px) {
        .glass-card, .weather-card, .metric-card, .knowledge-card {
            padding: 0.5rem !important;
            margin: 0.25rem 0 !important;
            border-radius: 8px !important;
        }
        h1, h2, h3, h4 {
            font-size: 1em !important;
        }
        .grid-container {
            grid-template-columns: 1fr !important;
            gap: 0.25rem !important;
        }
        table, th, td {
            font-size: 0.9em !important;
        }
    }
    /* Make tables scrollable on small screens */
    table {
        display: block;
        overflow-x: auto;
        width: 100%;
    }
    /* Responsive text */
    body, .main-bg {
        font-size: 1rem;
    }
    </style>
    """
