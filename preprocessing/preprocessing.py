import cv2
import numpy as np


def apply_gaussian_filter(image, kernel_size=(5, 5)):
    """
    Apply Gaussian Blur to reduce noise.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred


def apply_median_filter(image, kernel_size=5):
    """
    Apply Median Filter to remove salt-and-pepper noise.
    """
    median = cv2.medianBlur(image, kernel_size)
    return median


def resize_image(image, size=(224, 224)):
    """
    Resize image to a fixed size.
    """
    resized = cv2.resize(image, size)
    return resized


def normalize_image(image):
    """
    Normalize image to range [0, 1].
    """
    normalized = image / 255.0
    return normalized


def calculate_mse(image1, image2):
    """
    Calculate Mean Squared Error.
    """
    mse = np.mean((image1 - image2) ** 2)
    return mse


def calculate_psnr(image1, image2):
    """
    Calculate PSNR.
    """
    mse = calculate_mse(image1, image2)

    if mse == 0:
        return float("inf")

    pixel_max = 255.0
    psnr = 20 * np.log10(pixel_max / np.sqrt(mse))

    return psnr
