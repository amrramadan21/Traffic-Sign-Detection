import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROI_ROOT = PROJECT_ROOT / "data" / "detection" / "kmeans" / "rois"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "features"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class SIFTFeatureExtractor:
    def __init__(self, n_features=0):
        # Initialize the SIFT detector
        self.sift = cv2.SIFT_create(nfeatures=n_features)

    def extract_features(self, image_path: Path):
        """
        Loads an image and extracts SIFT keypoints and descriptors.
        """
        # Load ROI in grayscale (SIFT requires 1 channel)
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        return keypoints, descriptors

    def get_fixed_feature_vector(self, descriptors, vector_size=10):
        """
        Since different images have different numbers of keypoints,
        flatten the first 'vector_size' descriptors to create a fixed-length
        input for a classifier.
        """
        target_len = vector_size * 128
        if descriptors is None:
            return np.zeros(target_len, dtype=np.float32)

        flat_features = descriptors.flatten().astype(np.float32)
        if len(flat_features) >= target_len:
            return flat_features[:target_len]
        return np.pad(flat_features, (0, target_len - len(flat_features)))


def discover_roi_images(roi_root: Path):
    """
    Find ROI images. Supports both:
    - data/output/<image_name>/rois/*.png
    - any folder directly containing ROI image files
    """
    if roi_root.is_file() and roi_root.suffix.lower() in IMAGE_EXTENSIONS:
        return [roi_root]

    if not roi_root.exists():
        return []

    return sorted(
        path
        for path in roi_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def keypoints_to_records(keypoints):
    """
    Convert OpenCV KeyPoint objects to JSON-serializable dictionaries.
    """
    return [
        {
            "x": float(kp.pt[0]),
            "y": float(kp.pt[1]),
            "size": float(kp.size),
            "angle": float(kp.angle),
            "response": float(kp.response),
            "octave": int(kp.octave),
            "class_id": int(kp.class_id),
        }
        for kp in keypoints
    ]


def safe_output_stem(roi_path: Path, roi_root: Path):
    """
    Build a stable filename from the ROI's relative path.
    """
    try:
        relative = roi_path.relative_to(roi_root)
    except ValueError:
        relative = roi_path.name
    return str(relative).replace("\\", "__").replace("/", "__").replace(".", "_")


def process_all_rois(roi_root: Path, output_root: Path, vector_size: int):
    """
    Extract and save feature vectors, keypoints, and descriptors for all ROIs.
    """
    roi_paths = discover_roi_images(roi_root)
    if not roi_paths:
        print(f"No ROI images found under: {roi_root}")
        print("Run run_detection.py first, or pass a ROI folder/path:")
        print("python features/sift_extractor.py --roi-root data/output")
        return

    vectors_dir = output_root / "vectors"
    keypoints_dir = output_root / "keypoints"
    descriptors_dir = output_root / "descriptors"
    for directory in (vectors_dir, keypoints_dir, descriptors_dir):
        directory.mkdir(parents=True, exist_ok=True)

    extractor = SIFTFeatureExtractor()
    feature_matrix = []
    summary_rows = []

    for index, roi_path in enumerate(roi_paths):
        keypoints, descriptors = extractor.extract_features(roi_path)
        keypoints = keypoints or []
        vector = extractor.get_fixed_feature_vector(descriptors, vector_size)
        output_stem = safe_output_stem(roi_path, roi_root)

        np.save(vectors_dir / f"{output_stem}_vector.npy", vector)
        np.save(descriptors_dir / f"{output_stem}_descriptors.npy", descriptors)

        keypoint_records = keypoints_to_records(keypoints)
        with open(keypoints_dir / f"{output_stem}_keypoints.json", "w", encoding="utf-8") as file:
            json.dump(keypoint_records, file, indent=2)

        feature_matrix.append(vector)
        summary_rows.append(
            {
                "roi_index": index,
                "roi_path": str(roi_path),
                "feature_vector_file": str(vectors_dir / f"{output_stem}_vector.npy"),
                "keypoints_file": str(keypoints_dir / f"{output_stem}_keypoints.json"),
                "descriptors_file": str(descriptors_dir / f"{output_stem}_descriptors.npy"),
                "num_keypoints": len(keypoints),
                "descriptor_shape": "" if descriptors is None else str(descriptors.shape),
            }
        )

        print(f"[{index + 1}/{len(roi_paths)}] {roi_path.name}: {len(keypoints)} keypoints")

    feature_matrix = np.vstack(feature_matrix).astype(np.float32)
    np.save(output_root / "all_feature_vectors.npy", feature_matrix)

    with open(output_root / "summary.csv", "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "roi_index",
            "roi_path",
            "feature_vector_file",
            "keypoints_file",
            "descriptors_file",
            "num_keypoints",
            "descriptor_shape",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(output_root / "all_feature_vectors.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["roi_index", "roi_path"] + [f"f{i}" for i in range(feature_matrix.shape[1])])
        for row, vector in zip(summary_rows, feature_matrix):
            writer.writerow([row["roi_index"], row["roi_path"]] + vector.tolist())

    print("\nDone.")
    print(f"ROIs processed: {len(roi_paths)}")
    print(f"Feature matrix: {feature_matrix.shape}")
    print(f"Output folder : {output_root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract SIFT features from all ROI images.")
    parser.add_argument(
        "--roi-root",
        type=Path,
        default=DEFAULT_ROI_ROOT,
        help="ROI root folder or a single ROI image path. Default: data/output",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Folder where SIFT outputs will be saved. Default: data/features",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=10,
        help="Number of SIFT descriptors to flatten per ROI. Default: 10",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all_rois(args.roi_root, args.output_root, args.vector_size)