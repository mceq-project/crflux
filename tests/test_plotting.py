import os
from crflux.plotting import generate_model_comparison_plots


def test_plotting():
    """Test that generates plots for documentation and validation."""
    # Get the directory where this test file is located
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate plots using the shared plotting function
    generate_model_comparison_plots(save_dir=test_dir)

