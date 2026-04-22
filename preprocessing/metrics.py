import numpy as np
import cv2

def calculate_mse(image1, image2):
    """
    Calculate Mean Squared Error
    """
    mse = np.mean((image1 - image2) ** 2)
    return mse


def calculate_psnr(image1, image2):
    """
    Calculate PSNR
    """
    mse = calculate_mse(image1, image2)
    
    if mse == 0:
        return float('inf')
    
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    
    return psnr