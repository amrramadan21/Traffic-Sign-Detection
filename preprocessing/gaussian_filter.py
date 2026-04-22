import cv2

def apply_gaussian_filter(image, kernel_size=(5,5)):
    """
    Apply Gaussian Blur to reduce noise
    """
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred