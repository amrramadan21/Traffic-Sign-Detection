import os
import cv2

from utils.image_loader import load_image
from preprocessing.preprocessing import (
    apply_gaussian_filter,
    apply_median_filter,
    normalize_image,
    resize_image,
)
from utils.visualization import compare_filters  

FILTER_TYPE = "gaussian" and "median" 

input_folder = "data/raw/images"
output_folder = f"data/processed/{FILTER_TYPE}"

os.makedirs(output_folder, exist_ok=True)

images = os.listdir(input_folder)

print(f"Using filter: {FILTER_TYPE}")
print(f"Total images: {len(images)}")

test_image_path = os.path.join(input_folder, images[555])
test_image = load_image(test_image_path)

gaussian_test = apply_gaussian_filter(test_image)
median_test = apply_median_filter(test_image)

compare_filters(test_image, gaussian_test, median_test)

for i, img_name in enumerate(images):

    try:
        path = os.path.join(input_folder, img_name)

        image = load_image(path)

        if FILTER_TYPE == "gaussian":
            image = apply_gaussian_filter(image)

        elif FILTER_TYPE == "median":
            image = apply_median_filter(image)

        else:
            raise ValueError("Invalid filter type")

        image = resize_image(image)
        image = normalize_image(image)

        # save
        image = (image * 255).astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        save_path = os.path.join(output_folder, img_name)
        cv2.imwrite(save_path, image)

        if i % 100 == 0:
            print(f"Processed {i} images")

    except Exception as e:
        print(f"Error in {img_name}: {e}")

print(f"Done using {FILTER_TYPE} filter!")