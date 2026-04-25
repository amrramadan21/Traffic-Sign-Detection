# Detection — K-Means Batch Runner

This folder contains a K-Means based segmentation prototype and a batch runner
that processes the dataset to produce annotated images, per-ROI crops and
simple benchmark metrics.

This README explains how to run the code, what each script does, how outputs
are structured, and practical tuning / troubleshooting advice.

---

**Quick Links**
- Script: `detection/run_kmeans_batch.py`
- Notebook (exploration): `detection/kmeans_segmentation_prototype.ipynb`

---

**Requirements**
- Python 3.8+ (recommended)
- See top-level `requirements.txt` and install with:

```bash
pip install -r requirements.txt
```

This project primarily uses `opencv-python`, `numpy` (and `matplotlib` in the
notebook). The runner is cross-platform; it avoids Unix shebangs and autodetects
the project root on Windows.

---

**Quick Start**

Process a handful of images for quick verification:

```bash
python detection/run_kmeans_batch.py --max-images 10
```

Process the full dataset (defaults auto-detect `data/raw` and writes to
`data/detection/kmeans`):

```bash
python detection/run_kmeans_batch.py
```

Recommended accuracy-focused run (tries LAB color space and small morphology):

```bash
python detection/run_kmeans_batch.py --color-space lab --k 6 --morph-kernel 5 --dilate-iters 1
```

Use per-image variant evaluation (slow):

```bash
python detection/run_kmeans_batch.py --select-by-detection
```

---

**CLI Options (summary)**
- `--data-dir`: Path to `data/raw` containing `images/` and `annotations/`.
- `--out-dir`: Output base (defaults to `PROJECT_ROOT/data/detection/kmeans`).
- `--k`: Number of clusters for K-means (default 5). Try 5–8 for traffic signs.
- `--coverage-thresh`: Fractional coverage inside a ground-truth bbox to
  consider a cluster as covering the object (default 0.02).
- `--max-images`: Process at most N images (0 = all).
- `--use-processed` / `--no-use-processed`: Whether to prefer images in
  `data/processed/gaussian` or `data/processed/median` (default on).
- `--select-by-detection`: (Slow) run K-means on raw/gaussian/median and
  pick the variant that yields best detection metrics for each image.
- `--color-space`: `rgb`, `lab`, or `hsv` (default `rgb`). LAB often helps
  separate color channels for signs in varied lighting.
- `--morph-kernel`: Morphological kernel size used for postprocessing masks.
- `--dilate-iters`: Extra dilation on masks before measuring coverage.

Use `python detection/run_kmeans_batch.py --help` for the full help text.

---

How the code works (high-level)
- `find_project_root()` — finds the project root by locating `data/raw` or the
  `detection/detection.py` sentinel, enabling Windows-friendly invocation from
  any working directory.
- `load_image_bgr_to_rgb()` — loads images with OpenCV and returns RGB arrays
  for consistent visualization and cropping.
- `prepare_image_for_kmeans(image, color_space)` — converts an RGB image
  to the chosen color space (RGB / LAB / HSV) for clustering.
- `reshape_image_to_kmeans_samples()` — flattens an HxWx3 image into an N x 3
  float32 array for `cv2.kmeans`.
- `run_kmeans_rgb()` — runs OpenCV K-Means and returns `labels` and `centers`.
- `compute_centers_rgb()` — when clustering in LAB/HSV, cluster centers are
  recomputed in raw RGB using the raw pixels assigned to each label. This keeps
  visualization colors realistic even when clustering in a different color
  space.
- `cluster_masks_from_labels()` — creates one binary mask per cluster label.
- `postprocess_mask()` — morphological closing/opening and optional dilation to
  reduce noise and small fragments that cause false positive coverage.
- `parse_voc_annotation()` — simple VOC XML parser to extract `object` names
  and bounding boxes (xmin,ymin,xmax,ymax).
- `canonicalize_label()` — maps raw annotation names into one of the
  canonical classes: `Speedlimits`, `Stop sign`, `Traffic Lights`, `Cross Walk`.
- `choose_best_processed_image()` — quick heuristic: selects gaussian vs
  median processed variant via Laplacian variance (sharpness) when both exist.
- `evaluate_image_variant()` / `choose_best_variant_by_evaluation()` — slower
  evaluation path: run k-means on each variant (raw/gaussian/median), compute
  per-annotation coverage, and pick the variant that detects the most
  annotated objects (tie-breaker: summed coverage -> Laplacian). Useful when
  processed versions vary in quality per-image.
- `process_image(raw_image_path, proc_image_path, ...)` — main per-image
  pipeline. Important details:
  - segmentation is performed on the *processed* image (if provided)
    using the selected color space and k.
  - ROIs are always cropped from the *raw* image (annotation coords are
    aligned to the raw image). This avoids misaligned ROIs when processed
    images have different sizes or transforms.
  - masks are post-processed and coverage within annotated bboxes is computed
    to determine whether a cluster covers the object.
  - outputs: annotated image (original with boxes/coverage), roi crops
    (raw imagery), and a per-image JSON entry describing detections and timings.

---

Output structure
- `data/detection/kmeans/annotated/` — annotated original images with boxes + coverage.
- `data/detection/kmeans/rois/` — cropped ROIs (from the raw image) named
  `{image}_{obj_index}_{canonical_label}.png`.
- `data/detection/kmeans/per_image.json` — array of per-image dicts with fields:
  - `image`: original image filename
  - `processed_variant`: which variant was used (`raw`/`gaussian`/`median`)
  - `n_objects`: number of annotated objects
  - `detections`: list of detected clusters (name, best_cluster, coverage, bbox, detected)
  - `rois`: list of saved ROI metadata with `roi_path`, `coverage`, `detected`
  - `timings`: timing information (kmeans_seconds)
- `data/detection/kmeans/metrics.csv` — one-row-per-image summary (image, n_objects, n_detections, kmeans_seconds)
- `data/detection/kmeans/metrics_summary.json` — aggregated summary (counts, per-class rates, timing stats)

Note: generated dataset outputs under `data/` are ignored by the repository `.gitignore` by default.

---

Practical tuning advice
- Color space: start with `--color-space lab` for improved color-separation.
- `--k`: 5 is a sensible default (road, sky, dark/shadow, sign interior, sign border),
  but try 6–8 for complex scenes.
- Morphology: small `--morph-kernel` (3–7) removes speckle noise; set
  `--dilate-iters 1` to slightly enlarge masks when signs are thin.
- If processed images are available, `--use-processed` helps; use
  `--select-by-detection` to pick the best per-image (more CPU work).
- Filter small connected components (not currently automatic) to avoid
  small speckle artifacts increasing coverage — see `Next steps` below.

---

Known limitations and next steps
- K-Means is a simple, unsupervised approach. Cluster indices are not
  semantic and results can vary between runs. It struggles with heavy
  occlusion, extreme shadows or very small signs.
- Recommended improvements you may want to add:
  - Connected-component filtering on masks (keep only components above an
    area threshold or the largest component in the bbox) before measuring coverage.
  - `--save-masks` flag to persist cluster masks for a small subset for
    manual debugging (disabled by default to avoid too many files).
  - Parallelize processing with `concurrent.futures.ProcessPoolExecutor` for
    large datasets.
  - Replace heuristic detection with a small CNN to classify ROIs (best
    long-term solution).
    
---

File map (inside `detection/`):
- `run_kmeans_batch.py` — main batch runner (CLI)
- `kmeans_segmentation_prototype.ipynb` — exploratory notebook for visual
  debugging and experimentation

---

Contact / Notes
If something in the outputs looks wrong (misaligned ROIs or surprising
coverage numbers), try the `--color-space lab --morph-kernel 5 --dilate-iters 1`
command above, or run `--select-by-detection` on a small sample to see which
variant the script picks.

---

Happy to adapt this README (more detail, code references, examples) if you
want deeper walkthroughs of any specific function.
