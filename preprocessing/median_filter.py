import cv2

def apply_median_filter(image, kernel_size=5):
    """
    Apply Median Filter to remove salt-and-pepper noise
    """
    median = cv2.medianBlur(image, kernel_size)
    return median