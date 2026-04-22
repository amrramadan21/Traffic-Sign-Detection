import cv2

def load_image(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image