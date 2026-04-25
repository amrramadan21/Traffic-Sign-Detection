"""
Run K-Means segmentation across the raw dataset, save annotated images,
ROIs and produce simple benchmarks/metrics.

Creates outputs under `data/detection/kmeans/`:
 - annotated/: original images with bbox overlays and coverage
 - rois/: cropped ROIs per annotation
 - metrics.csv / metrics_summary.json / per_image.json

Usage:
    python run_kmeans_batch.py --max-images 10

Notes:
 - On Windows VS Code, the `#!/usr/bin/env python3` shebang may cause the
   runner to try invoking a Unix path which doesn't exist. This script now
   auto-detects the project root and uses Windows-friendly path resolution.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np


CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)


def load_image_bgr_to_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def reshape_image_to_kmeans_samples(image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected a 3-channel color image (H, W, 3).")
    h, w, c = image_rgb.shape
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    return pixels, (h, w, c)


def run_kmeans_rgb(pixels: np.ndarray, K: int = 5) -> tuple[np.ndarray, np.ndarray]:
    _compactness, labels, centers = cv2.kmeans(
        pixels, K, None, CRITERIA, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    return labels, centers


def reconstruct_quantized_image(labels: np.ndarray, centers: np.ndarray, shape_hwc: tuple[int, int, int]) -> np.ndarray:
    h, w, c = shape_hwc
    centers_ = centers.reshape(-1, 3)
    flat = labels.reshape(-1)
    quantized = centers_[flat].reshape(h, w, c)
    return np.clip(quantized, 0, 255).astype(np.uint8)


def cluster_masks_from_labels(labels: np.ndarray, shape_hw: tuple[int, int], K: int) -> list[np.ndarray]:
    h, w = shape_hw
    labels_2d = labels.reshape(h, w)
    return [(labels_2d == k).astype(np.uint8) * 255 for k in range(K)]


def parse_voc_annotation(xml_path: Path) -> list:
    import xml.etree.ElementTree as ET

    xml_path = str(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bnd = obj.find('bndbox')
        xmin = int(float(bnd.find('xmin').text))
        ymin = int(float(bnd.find('ymin').text))
        xmax = int(float(bnd.find('xmax').text))
        ymax = int(float(bnd.find('ymax').text))
        objs.append({'name': name, 'bbox': (xmin, ymin, xmax, ymax)})
    return objs


def canonicalize_label(name: str) -> str:
    n = name.lower() if name else ''
    if 'speed' in n or 'limit' in n:
        return 'Speedlimits'
    if 'stop' in n:
        return 'Stop sign'
    if 'light' in n:
        return 'Traffic Lights'
    if 'cross' in n or 'walk' in n:
        return 'Cross Walk'
    return 'Other'


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards to find the project root containing `data/raw` or `detection/detection.py`.

    Returns the first parent directory that looks like the project root, or the
    parent of this file as a sensible fallback.
    """
    if start is None:
        start = Path(__file__).resolve()
    p = start.resolve()
    for _ in range(8):
        if (p / 'data' / 'raw' / 'images').is_dir() or (p / 'data' / 'raw' / 'annotations').is_dir() or (p / 'detection' / 'detection.py').is_file():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parent


def _laplacian_variance_for_path(p: Path) -> float:
    """Return variance of the Laplacian (sharpness) for the given image path.

    A higher value indicates a sharper image (more edges). Used to choose
    between processed variants (gaussian vs median) when both are available.
    """
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return float(lap.var())


def choose_best_processed_image(original_path: Path, gaussian_path: Path, median_path: Path) -> tuple[Path, str]:
    """Choose which image to use: gaussian, median or raw.

    Strategy: prefer the available processed image with higher Laplacian
    variance (sharper/more edges). If neither processed image exists,
    fallback to the original `raw` image.
    Returns (selected_path, variant) where variant is one of
    'gaussian', 'median', or 'raw'.
    """
    g_exists = gaussian_path.is_file()
    m_exists = median_path.is_file()
    if g_exists and m_exists:
        g_score = _laplacian_variance_for_path(gaussian_path)
        m_score = _laplacian_variance_for_path(median_path)
        if g_score >= m_score:
            return gaussian_path, 'gaussian'
        return median_path, 'median'
    if g_exists:
        return gaussian_path, 'gaussian'
    if m_exists:
        return median_path, 'median'
    return original_path, 'raw'


def _image_shape(p: Path) -> tuple[int, int] | None:
    img = cv2.imread(str(p), cv2.IMREAD_ANYCOLOR)
    if img is None:
        return None
    return img.shape[0], img.shape[1]


def prepare_image_for_kmeans(image_rgb: np.ndarray, color_space: str) -> np.ndarray:
    """Convert RGB image to the chosen color space for k-means."""
    cs = color_space.lower()
    if cs == 'rgb':
        return image_rgb
    if cs == 'lab':
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    if cs == 'hsv':
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    raise ValueError(f'Unsupported color space: {color_space}')


def compute_centers_rgb(labels: np.ndarray, original_pixels: np.ndarray, K: int) -> np.ndarray:
    """Compute cluster centers in RGB space using original pixels and labels."""
    flat = labels.reshape(-1)
    centers = np.zeros((K, 3), dtype=np.float32)
    for k in range(K):
        mask = (flat == k)
        if mask.sum() > 0:
            centers[k] = original_pixels[mask].mean(axis=0)
        else:
            centers[k] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return centers


def postprocess_mask(mask: np.ndarray, morph_kernel: int = 5, dilate_iters: int = 0) -> np.ndarray:
    if morph_kernel and morph_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if dilate_iters and dilate_iters > 0:
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel2, iterations=dilate_iters)
    return mask


def evaluate_image_variant(image_path: Path, ann_path: Path | None, K: int, coverage_thresh: float, color_space: str = 'rgb', morph_kernel: int = 5, dilate_iters: int = 0) -> dict:
    """Run k-means on `image_path` and evaluate detection coverage against annotations.

    Returns a dict with keys: 'n_detected' (int) and 'sum_cov' (float).
    Applies morphological cleanup and optional dilation before measuring coverage.
    """
    try:
        img_rgb = load_image_bgr_to_rgb(image_path)
    except FileNotFoundError:
        return {'n_detected': 0, 'sum_cov': 0.0}
    orig_pixels, shape_hwc = reshape_image_to_kmeans_samples(img_rgb)
    # prepare image for kmeans in the chosen color space
    img_k = prepare_image_for_kmeans(img_rgb, color_space)
    pixels_k, shape_hwc2 = reshape_image_to_kmeans_samples(img_k)
    h, w, _ = shape_hwc
    labels, _ = run_kmeans_rgb(pixels_k, K)
    masks = cluster_masks_from_labels(labels, (h, w), K)
    # postprocess masks
    masks = [postprocess_mask(m, morph_kernel=morph_kernel, dilate_iters=dilate_iters) for m in masks]

    if ann_path and ann_path.is_file():
        objects = parse_voc_annotation(ann_path)
    else:
        objects = []

    n_detected = 0
    sum_cov = 0.0
    for obj in objects:
        xmin, ymin, xmax, ymax = obj['bbox']
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(0, min(xmax, w - 1))
        ymax = max(0, min(ymax, h - 1))
        bbox_area = max(1, (xmax - xmin) * (ymax - ymin))
        best_cov = 0.0
        for mask in masks:
            roi = mask[ymin:ymax, xmin:xmax]
            cov = float((roi > 0).sum()) / bbox_area
            if cov > best_cov:
                best_cov = cov
        sum_cov += best_cov
        if best_cov > coverage_thresh:
            n_detected += 1

    return {'n_detected': n_detected, 'sum_cov': float(sum_cov)}


def choose_best_variant_by_evaluation(raw_path: Path, gaussian_path: Path, median_path: Path, ann_path: Path | None, K: int, coverage_thresh: float, color_space: str = 'rgb', morph_kernel: int = 5, dilate_iters: int = 0) -> tuple[Path, str]:
    """Evaluate raw/gaussian/median variants by running k-means and choose the best.

    Ranking order: highest `n_detected` (primary), then highest `sum_cov` (secondary).
    Falls back to Laplacian sharpness if tied.
    """
    candidates = []
    variants = [('raw', raw_path), ('gaussian', gaussian_path), ('median', median_path)]
    raw_shape = _image_shape(raw_path)
    for name, p in variants:
        if not p.is_file():
            continue
        # skip processed images that don't match the raw image shape
        p_shape = _image_shape(p)
        if raw_shape is not None and p_shape is not None and p_shape != raw_shape and name != 'raw':
            continue
        metrics = evaluate_image_variant(p, ann_path, K, coverage_thresh, color_space=color_space, morph_kernel=morph_kernel, dilate_iters=dilate_iters)
        candidates.append((p, name, metrics))

    if not candidates:
        return raw_path, 'raw'

    # sort by n_detected then sum_cov
    candidates.sort(key=lambda x: (x[2]['n_detected'], x[2]['sum_cov']), reverse=True)
    top = candidates[0]
    # tie-breaker using Laplacian if necessary
    if len(candidates) > 1 and candidates[0][2]['n_detected'] == candidates[1][2]['n_detected'] and candidates[0][2]['sum_cov'] == candidates[1][2]['sum_cov']:
        # compute laplacian scores
        lap_scores = [(c[0], c[1], _laplacian_variance_for_path(c[0])) for c in candidates[:2]]
        lap_scores.sort(key=lambda x: x[2], reverse=True)
        return lap_scores[0][0], lap_scores[0][1]

    return top[0], top[1]


def process_image(
    raw_image_path: Path,
    proc_image_path: Path | None,
    ann_path: Path | None,
    out_annotated_dir: Path,
    out_rois_dir: Path,
    K: int,
    coverage_thresh: float,
    color_space: str = 'rgb',
    morph_kernel: int = 5,
    dilate_iters: int = 0,
):
    # load raw image for ROI cropping and overlay
    raw_rgb = load_image_bgr_to_rgb(raw_image_path)
    # load processed image for segmentation if provided; otherwise use raw
    if proc_image_path is not None and proc_image_path.is_file():
        proc_rgb = load_image_bgr_to_rgb(proc_image_path)
        # guard: if sizes mismatch, fallback to raw
        if proc_rgb.shape[:2] != raw_rgb.shape[:2]:
            proc_rgb = raw_rgb
    else:
        proc_rgb = raw_rgb

    # prepare pixels for k-means in desired color space
    img_k = prepare_image_for_kmeans(proc_rgb, color_space)
    pixels_k, shape_hwc = reshape_image_to_kmeans_samples(img_k)
    h, w, _ = shape_hwc
    original_pixels = raw_rgb.reshape(-1, 3).astype(np.float32)

    t0 = time.perf_counter()
    labels, centers_k = run_kmeans_rgb(pixels_k, K)
    t1 = time.perf_counter()

    # compute RGB centers for visualization
    centers_rgb = compute_centers_rgb(labels, original_pixels, K)
    segmented_rgb = reconstruct_quantized_image(labels, centers_rgb, shape_hwc)

    masks = cluster_masks_from_labels(labels, (h, w), K)
    masks = [postprocess_mask(m, morph_kernel=morph_kernel, dilate_iters=dilate_iters) for m in masks]

    stem = raw_image_path.stem

    if ann_path and ann_path.is_file():
        objects = parse_voc_annotation(ann_path)
    else:
        objects = []

    detection_results = []
    rois = []
    for idx, obj in enumerate(objects):
        xmin, ymin, xmax, ymax = obj['bbox']
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(0, min(xmax, w - 1))
        ymax = max(0, min(ymax, h - 1))
        bbox_area = max(1, (xmax - xmin) * (ymax - ymin))
        best_k = None
        best_cov = 0.0
        for k, mask in enumerate(masks):
            roi_mask = mask[ymin:ymax, xmin:xmax]
            cov = float((roi_mask > 0).sum()) / bbox_area
            if cov > best_cov:
                best_cov = cov
                best_k = k
        detected = best_cov > coverage_thresh
        detection_results.append({'name': obj['name'], 'best_cluster': best_k, 'coverage': best_cov, 'bbox': (xmin, ymin, xmax, ymax), 'detected': detected})

        # save ROI crop from the raw image (annotations are relative to raw)
        roi_img = raw_rgb[ymin:ymax, xmin:xmax]
        roi_path = out_rois_dir / f"{stem}_{idx}_{canonicalize_label(obj['name']).replace(' ', '_')}.png"
        roi_path.parent.mkdir(parents=True, exist_ok=True)
        if roi_img.size:
            cv2.imwrite(str(roi_path), cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(roi_path), np.zeros((1, 1, 3), dtype=np.uint8))
        rois.append({'name': obj['name'], 'canonical': canonicalize_label(obj['name']), 'bbox': (xmin, ymin, xmax, ymax), 'roi_path': str(roi_path), 'coverage': best_cov, 'detected': detected})

    # overlay boxes onto a BGR image for saving (from raw image)
    overlay_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
    for res in detection_results:
        x1, y1, x2, y2 = res['bbox']
        color = (0, 0, 255)  # red in BGR
        cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{res['name']}:{res['coverage']:.2f}"
        cv2.putText(overlay_bgr, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    annotated_path = out_annotated_dir / f"{stem}_annotated.png"
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(annotated_path), overlay_bgr)

    timings = {'kmeans_seconds': t1 - t0}

    per_image = {
        'image': str(raw_image_path.name),
        'n_objects': len(objects),
        'detections': detection_results,
        'rois': rois,
        'timings': timings,
        'annotated_image': str(annotated_path),
    }

    return per_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=None, help='Path to data/raw folder (contains images/ and annotations/)')
    parser.add_argument('--out-dir', type=Path, default=None, help='Output base folder (defaults to project_root/data/detection/kmeans)')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--coverage-thresh', type=float, default=0.02)
    parser.add_argument('--max-images', type=int, default=0, help='Process at most N images (0 = all)')
    parser.add_argument('--use-processed', action=argparse.BooleanOptionalAction, default=True,
                        help='Use processed images from data/processed (gaussian/median) when available.')
    parser.add_argument('--select-by-detection', action='store_true', default=False,
                        help='(Slow) Evaluate raw/gaussian/median by running k-means and choose the variant that yields best detection results.')
    parser.add_argument('--color-space', choices=['rgb', 'lab', 'hsv'], default='rgb', help='Color space for k-means clustering (rgb, lab, hsv).')
    parser.add_argument('--morph-kernel', type=int, default=5, help='Morphological kernel size for postprocessing masks (odd integer).')
    parser.add_argument('--dilate-iters', type=int, default=0, help='Dilation iterations on mask before measuring coverage.')
    args = parser.parse_args()

    # Auto-detect project root and set sensible defaults when arguments are not provided
    project_root = find_project_root(Path(__file__).resolve())
    if args.data_dir is None:
        args.data_dir = project_root / 'data' / 'raw'
    else:
        args.data_dir = args.data_dir.resolve()
    if args.out_dir is None:
        args.out_dir = project_root / 'data' / 'detection' / 'kmeans'
    else:
        args.out_dir = args.out_dir.resolve()

    images_dir = args.data_dir / 'images'
    ann_dir = args.data_dir / 'annotations'
    out_annotated_dir = args.out_dir / 'annotated'
    out_rois_dir = args.out_dir / 'rois'
    out_metrics_csv = args.out_dir / 'metrics.csv'
    out_metrics_json = args.out_dir / 'metrics_summary.json'
    out_per_image = args.out_dir / 'per_image.json'

    args.out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in images_dir.glob('*.png')])
    if args.max_images and args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    per_images = []
    all_objects = 0
    detected_objects = 0
    class_stats = {}
    timings = []

    processed_base = project_root / 'data' / 'processed'
    gauss_dir = processed_base / 'gaussian'
    med_dir = processed_base / 'median'

    for i, img_path in enumerate(image_paths, start=1):
        ann_path = ann_dir / (img_path.stem + '.xml')
        # pick the best processed variant if requested
        if args.use_processed:
            gauss_path = gauss_dir / img_path.name
            med_path = med_dir / img_path.name
            if args.select_by_detection:
                selected_path, variant = choose_best_variant_by_evaluation(img_path, gauss_path, med_path, ann_path if ann_path.exists() else None, args.k, args.coverage_thresh, color_space=args.color_space, morph_kernel=args.morph_kernel, dilate_iters=args.dilate_iters)
            else:
                selected_path, variant = choose_best_processed_image(img_path, gauss_path, med_path)
        else:
            selected_path, variant = img_path, 'raw'

        print(f"[{i}/{len(image_paths)}] Processing {img_path.name} (using {variant})")
        per_image = process_image(img_path, selected_path if selected_path != img_path else None, ann_path if ann_path.exists() else None, out_annotated_dir, out_rois_dir, args.k, args.coverage_thresh, color_space=args.color_space, morph_kernel=args.morph_kernel, dilate_iters=args.dilate_iters)
        # always record the original image name and which variant we used
        per_image['image'] = str(img_path.name)
        per_image['processed_variant'] = variant
        per_images.append(per_image)
        # aggregate
        n_objs = per_image['n_objects']
        all_objects += n_objs
        for d in per_image['detections']:
            canonical = canonicalize_label(d['name'])
            class_stats.setdefault(canonical, {'total': 0, 'detected': 0})
            class_stats[canonical]['total'] += 1
            if d.get('detected'):
                class_stats[canonical]['detected'] += 1
                detected_objects += 1
        timings.append(per_image['timings']['kmeans_seconds'])

    # write per-image JSON
    with open(out_per_image, 'w', encoding='utf8') as f:
        json.dump(per_images, f, indent=2)

    # write metrics CSV (one row per image)
    with open(out_metrics_csv, 'w', newline='', encoding='utf8') as f:
        fieldnames = ['image', 'n_objects', 'n_detections', 'kmeans_seconds']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in per_images:
            writer.writerow({'image': p['image'], 'n_objects': p['n_objects'], 'n_detections': sum(1 for d in p['detections'] if d.get('detected')), 'kmeans_seconds': p['timings']['kmeans_seconds']})

    # summary
    summary = {
        'n_images': len(image_paths),
        'n_annotations': all_objects,
        'n_detected_annotations': detected_objects,
        'detection_rate_over_annotations': (detected_objects / all_objects) if all_objects else None,
        'per_class': {k: {'total': v['total'], 'detected': v['detected'], 'detection_rate': (v['detected'] / v['total'] if v['total'] else None)} for k, v in class_stats.items()},
        'timing': {
            'count': len(timings),
            'avg_kmeans_s': mean(timings) if timings else None,
            'median_kmeans_s': median(timings) if timings else None,
            'min_kmeans_s': min(timings) if timings else None,
            'max_kmeans_s': max(timings) if timings else None,
        },
        'outputs': {
            'annotated_dir': str(out_annotated_dir),
            'rois_dir': str(out_rois_dir),
            'per_image_json': str(out_per_image),
            'metrics_csv': str(out_metrics_csv),
        },
    }

    with open(out_metrics_json, 'w', encoding='utf8') as f:
        json.dump(summary, f, indent=2)

    print('\nProcessing complete.')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
