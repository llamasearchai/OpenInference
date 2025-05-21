"""
OpenInference: High-Performance AI Inference Engine

A comprehensive, high-performance AI inference engine designed for efficient
model serving across a variety of hardware configurations.
"""

# Import version
__version__ = "0.1.0"

# Import main components
from .main import OpenInference
from .hardware.accelerator import AcceleratorType, HardwareManager
from .monitoring.performance_tracker import PerformanceTracker

# Set up logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

def get_version():
    return __version__

# Convenience function to create an inference engine
def create_inference_engine(**kwargs):
    """Create and initialize an OpenInference engine."""
    return OpenInference(**kwargs)