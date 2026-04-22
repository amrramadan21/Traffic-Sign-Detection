import cv2

def resize_image(image, size=(224, 224)):
    """
    Resize image to a fixed size
    """
    resized = cv2.resize(image, size)
    return resized