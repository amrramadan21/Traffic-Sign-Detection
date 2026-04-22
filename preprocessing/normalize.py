import numpy as np

def normalize_image(image):
    """
    Normalize image to range [0, 1]
    """
    normalized = image / 255.0
    return normalized