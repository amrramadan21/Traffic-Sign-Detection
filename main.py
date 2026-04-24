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

FILTER_TYPES = ["gaussian", "median"]

input_folder = "data/raw/images"

images = os.listdir(input_folder)
print(f"Total images: {len(images)}")

# Show a comparison of both filters on a sample image (requires a display).
if images:
    sample_idx = min(555, len(images) - 1)
    test_image = load_image(os.path.join(input_folder, images[sample_idx]))
    gaussian_test = apply_gaussian_filter(test_image)
    median_test   = apply_median_filter(test_image)
    compare_filters(test_image, gaussian_test, median_test)

for filter_type in FILTER_TYPES:
    output_folder = f"data/processed/{filter_type}"
    os.makedirs(output_folder, exist_ok=True)

    print(f"\nApplying {filter_type} filter...")

    for i, img_name in enumerate(images):
        try:
            path  = os.path.join(input_folder, img_name)
            image = load_image(path)

            if filter_type == "gaussian":
                image = apply_gaussian_filter(image)
            else:
                image = apply_median_filter(image)

            image = resize_image(image)
            image = normalize_image(image)

            image = (image * 255).astype("uint8")
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(output_folder, img_name), image)

            if i % 100 == 0:
                print(f"  {i}/{len(images)} processed")

        except Exception as e:
            print(f"  Error in {img_name}: {e}")

    print(f"Done: {filter_type} filter saved to {output_folder}/")

print("\nAll filters applied.")