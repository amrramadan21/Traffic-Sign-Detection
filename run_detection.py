"""
run_detection.py
================
Run the detection pipeline on every image in the processed dataset.

For each image, both the Gaussian-filtered and Median-filtered versions are
loaded and processed by detect_pipeline().  The version that produces the
higher-quality result (measured by detection confidence score) is chosen, and
its outputs are saved to data/output/.

Output layout
-------------
data/output/
├── <image_stem>/
│   ├── annotated.png      – original image with coloured bounding boxes
│   ├── mask_red.png       – binary mask for red regions
│   ├── mask_blue.png      – binary mask for blue regions
│   ├── mask_combined.png  – union of both colour masks
│   ├── filter_used.txt    – which filter won ("gaussian" or "median")
│   └── rois/
│       ├── roi_000.png    – cropped region-of-interest #0
│       └── ...
└── summary.csv            – per-image detection summary
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# Locate the project root so imports work however the script is invoked.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.detection import detect_pipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GAUSSIAN_DIR = PROJECT_ROOT / "data" / "processed" / "gaussian"
MEDIAN_DIR   = PROJECT_ROOT / "data" / "processed" / "median"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "output"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

CSV_FIELDS = [
    "image",
    "filter_chosen",
    "num_detections",
    "num_red",
    "num_blue",
    "num_traffic_lights",
    "num_rois",
    "total_score",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _list_images(directory: Path) -> set[str]:
    """Return the set of image filenames inside *directory*."""
    if not directory.is_dir():
        return set()
    return {
        f.name
        for f in directory.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    }


def _quality_score(result: dict) -> float:
    """
    Aggregate quality of a detection result.

    Primary key  : sum of per-detection confidence scores.
    Secondary key: total non-zero pixels in the combined colour mask
                   (scaled small so it only acts as a tiebreaker).
    """
    primary   = sum(d["score"] for d in result["detections"])
    secondary = float(np.count_nonzero(result["combined_mask"])) * 1e-5
    return primary + secondary


def _save_result(stem: str, result: dict, filter_name: str, out_root: Path) -> None:
    """Persist all outputs for one image under *out_root/<stem>/*."""
    img_dir = out_root / stem
    roi_dir = img_dir / "rois"
    img_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(img_dir / "annotated.png"),     result["output"])
    cv2.imwrite(str(img_dir / "mask_red.png"),      result["red_mask"])
    cv2.imwrite(str(img_dir / "mask_blue.png"),     result["blue_mask"])
    cv2.imwrite(str(img_dir / "mask_combined.png"), result["combined_mask"])

    (img_dir / "filter_used.txt").write_text(filter_name, encoding="utf-8")

    for i, roi in enumerate(result["rois"]):
        if roi is not None and roi.size > 0:
            cv2.imwrite(str(roi_dir / f"roi_{i:03d}.png"), roi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    gauss_names  = _list_images(GAUSSIAN_DIR)
    median_names = _list_images(MEDIAN_DIR)
    all_names    = sorted(gauss_names | median_names)

    if not all_names:
        print(
            "No processed images found.\n"
            "Expected locations:\n"
            f"  {GAUSSIAN_DIR}\n"
            f"  {MEDIAN_DIR}\n"
            "Run main.py (or preprocessing) first to populate those folders."
        )
        sys.exit(1)

    print(f"Found {len(all_names)} unique image(s) across both filter folders.")
    print(f"Output will be written to: {OUTPUT_DIR}\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    filter_counts = {"gaussian": 0, "median": 0}

    for idx, name in enumerate(all_names, 1):
        stem = Path(name).stem
        label = f"[{idx:>4}/{len(all_names)}] {name}"

        # ---- Run detection on each available filter variant ----
        candidates: dict[str, dict] = {}

        for filter_name, directory in [("gaussian", GAUSSIAN_DIR), ("median", MEDIAN_DIR)]:
            img_path = directory / name
            if not img_path.exists():
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  WARNING: could not read {img_path}, skipping.")
                continue

            try:
                candidates[filter_name] = detect_pipeline(image)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"  WARNING: detection failed for {filter_name}/{name}: {exc}")

        if not candidates:
            print(f"{label}  ->  SKIPPED (no readable variants)")
            continue

        # ---- Pick the better filter ----
        best_filter = max(candidates, key=lambda k: _quality_score(candidates[k]))
        best_result = candidates[best_filter]
        score       = _quality_score(best_result)
        n_det       = len(best_result["detections"])

        print(f"{label}  ->  {best_filter:8s}  detections={n_det}  score={score:.3f}")

        # ---- Persist outputs ----
        _save_result(stem, best_result, best_filter, OUTPUT_DIR)

        # ---- Accumulate summary ----
        det = best_result["detections"]
        filter_counts[best_filter] += 1
        summary_rows.append({
            "image":               name,
            "filter_chosen":       best_filter,
            "num_detections":      n_det,
            "num_red":             sum(1 for d in det if d["color"] == "Red"),
            "num_blue":            sum(1 for d in det if d["color"] == "Blue"),
            "num_traffic_lights":  sum(1 for d in det if d["color"] == "Traffic Light"),
            "num_rois":            len(best_result["rois"]),
            "total_score":         round(score, 4),
        })

    # ---- Write CSV ----
    csv_path = OUTPUT_DIR / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    # ---- Final report ----
    total_det  = sum(r["num_detections"]     for r in summary_rows)
    total_red  = sum(r["num_red"]            for r in summary_rows)
    total_blue = sum(r["num_blue"]           for r in summary_rows)
    total_tl   = sum(r["num_traffic_lights"] for r in summary_rows)

    print("\n" + "=" * 60)
    print(f"  Images processed  : {len(summary_rows)}")
    print(f"  Filter chosen     : gaussian={filter_counts['gaussian']}, "
          f"median={filter_counts['median']}")
    print(f"  Total detections  : {total_det}  "
          f"(red={total_red}, blue={total_blue}, traffic_lights={total_tl})")
    print(f"  Summary CSV       : {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
