import cv2
import numpy as np
import os


def convert_to_hsv(image):
    """
    Convert the input image from BGR to HSV.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def clean_mask(mask, color_name, small=False):
    """
    Remove small noise from a mask.
    When `small=True` (used for far-sign pass), use a tiny kernel so we
    don't erase the few pixels that small distant signs produce.

    Blue open kernel is 2×2 (not 4×4): pedestrian/direction signs have a
    narrow (~2–3 px) blue border around their white interior content.
    A 4×4 open erases those thin borders, fragmenting the ring into
    disconnected pieces that produce multiple spurious boxes.  A 2×2 open
    still removes 1-pixel sky noise while preserving narrow sign edges.
    """
    if small:
        open_kernel = np.ones((2, 2), np.uint8)
        close_kernel = np.ones((3, 3), np.uint8)
    elif color_name == "Blue":
        open_kernel = np.ones((2, 2), np.uint8)   # was 4×4
        close_kernel = np.ones((6, 6), np.uint8)
    else:
        open_kernel = np.ones((2, 2), np.uint8)
        close_kernel = np.ones((4, 4), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask


def create_color_masks(hsv, small=False):
    """
    Build red and blue masks in HSV space.
    Red uses two ranges because hue wraps around in HSV.

    Saturation thresholds are intentionally split:
    - Normal scale (small=False):
        Red  S ≥ 140 — bricks (S≈80-130) and golden-hour sky (S≈100-130)
             are now below the threshold; retroreflective signs (S≈150-255)
             are clearly above it.
        Blue S ≥ 110 — deep blue sky (S≈70-100) rejected; road-sign blue
             (S≈130-230) still detected.
    - Upscale pass (small=True):  S ≥ 70 — distant / faded / hazy signs.
    """
    sat_min_red  = 70 if small else 140
    sat_min_blue = 70 if small else 110

    lower_red1 = np.array([0,   sat_min_red, 50])
    upper_red1 = np.array([12,  255,         255])
    lower_red2 = np.array([168, sat_min_red, 50])
    upper_red2 = np.array([180, 255,         255])

    lower_blue = np.array([95, sat_min_blue, 40])
    upper_blue = np.array([130, 255,         255])

    red_mask_1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask_2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    red_mask = clean_mask(red_mask, "Red", small=small)
    blue_mask = clean_mask(blue_mask, "Blue", small=small)

    return red_mask, blue_mask


def classify_shape(contour):
    """
    Keep simple sign-like shapes: triangle, rectangle, circle-like.
    Now accepts 5-sided shapes too (some signs approximate to pentagons).
    """
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "rejected", 0

    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    sides = len(approx)

    if sides == 3:
        return "triangle", sides
    if sides == 4:
        return "rectangle", sides
    if sides >= 5:          # pentagon, hexagon, circle-like — all valid signs
        return "circle-like", sides

    return "rejected", sides


def touches_border(x, y, w, h, image_width, image_height, margin=5):
    """
    Large background regions often touch the image border.
    """
    return (
        x <= margin
        or y <= margin
        or (x + w) >= image_width - margin
        or (y + h) >= image_height - margin
    )


def calculate_iou(box_a, box_b):
    """
    Compute IoU overlap between two bounding boxes.
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)

    if left >= right or top >= bottom:
        return 0.0

    intersection = (right - left) * (bottom - top)
    union = (aw * ah) + (bw * bh) - intersection
    return intersection / float(union)


def calculate_containment(box_a, box_b):
    """
    Return the fraction of box_a that is inside box_b.
    Used to collapse a small detection fully contained in a larger one.
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)

    if left >= right or top >= bottom:
        return 0.0

    intersection = (right - left) * (bottom - top)
    return intersection / float(aw * ah)


def remove_overlapping_detections(detections, iou_threshold=0.25):
    """
    Suppress duplicate boxes using NMS.

    Rules (applied in priority order):
    1. IoU > threshold → suppress lower-scored detection (cross-color aware).
    2. Containment > threshold → suppress the smaller box.
       - Normal pair: containment threshold = 0.70.
       - Red/Blue inside a Traffic Light box: threshold = 0.40.
         A light bulb needs only 40 % overlap with the TL housing to be
         considered the same object — prevents the active red/green light
         from also appearing as a standalone "Red" sign detection.
    3. Traffic Light detections carry inflated scores so they are always
       processed before the plain Red detections they contain.
    """
    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    filtered = []

    for detection in detections:
        keep = True

        for saved in filtered:
            overlap = calculate_iou(detection["bbox"], saved["bbox"])
            contained = calculate_containment(detection["bbox"], saved["bbox"])

            if overlap > iou_threshold:
                keep = False
                break

            # Decide the containment threshold based on the saved class
            if saved["color"] == "Traffic Light" and detection["color"] in ("Red", "Blue"):
                contain_thresh = 0.40   # aggressively suppress light-bulb detections
            else:
                contain_thresh = 0.70

            if contained > contain_thresh:
                keep = False
                break

        if keep:
            filtered.append(detection)

    return filtered


def build_detection(contour, mask, image, color_name, min_area, max_area_ratio):
    """
    Convert one contour into a detection dict if it passes all filters.
    """
    image_height, image_width = image.shape[:2]
    image_area = image_height * image_width
    area = cv2.contourArea(contour)

    if area < min_area or area > max_area_ratio * image_area:
        return None

    x, y, w, h = cv2.boundingRect(contour)

    # Minimum pixel size — relaxed to 8 px to catch far signs
    if w < 8 or h < 8:
        return None

    # Large signs (> 2 % of image) are valid even at frame edges; only
    # filter border-touching regions for small/medium detections.
    is_large = area > 0.02 * image_area

    # --- UPDATED: Border Rejection ---
    # Apply border rejection to ALL small/medium detections.
    # Clouds bleed off the edges of the frame; real signs usually sit fully inside.
    if not is_large:
        if touches_border(x, y, w, h, image_width, image_height):
            return None

    # Sky filter — two tiers:
    if color_name == "Blue":
        if not is_large and y < int(0.08 * image_height) and area > 0.008 * image_area:
            return None
        if is_large and y < int(0.05 * image_height):
            return None

    # --- NEW: Sky Filter for Red ---
    if color_name == "Red":
        if not is_large and y < int(0.08 * image_height) and area > 0.005 * image_area:
            return None
        if is_large and y < int(0.05 * image_height):
            return None

    aspect_ratio = w / float(h)
    if color_name == "Blue":
        # Wider range to accept landscape road signs and portrait pedestrian signs
        if aspect_ratio < 0.35 or aspect_ratio > 2.50:
            return None
    elif color_name == "Red":
        # --- UPDATED: TIGHTENED ASPECT RATIO ---
        # Red signs (Stop, Yield, Speed Limit) are symmetrical (~1:1).
        # Sunset clouds form wide, stretched horizontal bands.
        if aspect_ratio < 0.65 or aspect_ratio > 1.35:
            return None

    # --- NEW: Solidity Check ---
    # Solidity = Contour Area / Convex Hull Area
    # Signs are rigid geometric shapes. Clouds are amorphous and jagged.
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        solidity = area / float(hull_area)
        if solidity < 0.75:  # Reject anything that isn't a solid, convex shape
            return None

    bbox_area = float(w * h)
    fill_ratio = area / bbox_area
    
    # --- UPDATED: Tighter Density Checks for Red ---
    if color_name == "Blue":
        min_fill_ratio = 0.15
        min_color_ratio = 0.15
    elif color_name == "Red":
        min_fill_ratio = 0.40   # Increased to enforce solid shapes
        min_color_ratio = 0.30  # Increased to enforce solid color
    else:
        min_fill_ratio = 0.08
        min_color_ratio = 0.06

    if fill_ratio < min_fill_ratio:
        return None

    roi_mask = mask[y: y + h, x: x + w]
    color_pixels = cv2.countNonZero(roi_mask)
    color_ratio = color_pixels / bbox_area

    if color_ratio < min_color_ratio:
        return None

    shape_name, shape_sides = classify_shape(contour)
    if shape_name == "rejected":
        return None

    score = area + (1000 * color_ratio) + (200 * fill_ratio)

    return {
        "color": color_name,
        "bbox": (x, y, w, h),
        "area": area,
        "fill_ratio": fill_ratio,
        "color_ratio": color_ratio,
        "shape_name": shape_name,
        "shape_sides": shape_sides,
        "score": score,
    }

def detect_red_circles(image, red_mask):
    """
    Recover circular red signs (e.g. speed-limit rings) with Hough circles.

    Stricter than before:
    - param2 raised from 18 -> 28 to reduce false circles.
    - color_ratio threshold raised from 0.08 -> 0.15.
    - Circles touching the image border are rejected.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)
    image_height, image_width = image.shape[:2]

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=100,
        param2=22,   # 28→22: speed-limit rings are often blurred by preprocessing;
                     # looser accumulator still rejects non-circular shapes
        minRadius=8,
        maxRadius=100,
    )

    if circles is None:
        return []

    detections = []

    for center_x, center_y, radius in np.round(circles[0]).astype(int):
        x = max(center_x - radius, 0)
        y = max(center_y - radius, 0)
        w = min(2 * radius, image_width - x)
        h = min(2 * radius, image_height - y)

        if w < 12 or h < 12:
            continue

        # Skip circles that touch the image border (likely background arcs)
        if touches_border(x, y, w, h, image_width, image_height, margin=8):
            continue

        ring_mask = np.zeros_like(red_mask)
        cv2.circle(ring_mask, (center_x, center_y), radius, 255, 3)
        overlap_mask = cv2.bitwise_and(red_mask, ring_mask)

        ring_pixels = cv2.countNonZero(ring_mask)
        red_pixels = cv2.countNonZero(overlap_mask)
        color_ratio = red_pixels / float(ring_pixels) if ring_pixels else 0.0

        if color_ratio < 0.10:   # 0.15→0.10: thin speed-limit rings have low
            continue               # overlap with the 3-px Hough sampling band

        detections.append(
            {
                "color": "Red",
                "bbox": (x, y, w, h),
                "area": float(np.pi * radius * radius),
                "fill_ratio": color_ratio,
                "color_ratio": color_ratio,
                "shape_name": "circle-like",
                "shape_sides": 8,
                "score": (np.pi * radius * radius) + (1200 * color_ratio),
            }
        )

    return detections


def detect_traffic_lights(image, hsv):
    """
    Detect traffic lights by finding their bright colored light blobs.

    Previous approach (find dark housing first) failed because real-world
    housings appear as mid-gray (60-100), not near-black, so the dark mask
    was nearly empty.

    New strategy — lights first, housing inferred:
    1. Build tight color masks for bright red / yellow / green blobs.
       High Value (>=150) requirement selects self-emitting light sources
       over reflective sign surfaces.
    2. For each blob, compute the brightness contrast between the blob ROI
       and its immediate surroundings.  A glowing bulb is notably brighter
       than its housing; a painted sign is not.
    3. Group blobs that share the same horizontal position but are stacked
       vertically — the classic traffic-light arrangement.
    4. Pad each group's bounding box to cover the full housing and report
       it as a "Traffic Light" detection with an inflated score so NMS
       suppresses any plain "Red" detection of the same light bulb.
    """
    image_height, image_width = image.shape[:2]
    image_area = image_height * image_width
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Red/yellow: V ≥ 120 — bright enough in all conditions.
    # Green: V ≥ 85 (was 70).  Lowering to 70 caught dim overcast bulbs
    # but also warm autumn foliage, producing whole-scene false positives.
    # 85 is still below the typical daytime bulb (V≈100-150) while staying
    # above most shade-leaf noise.  S ≥ 60 unchanged.
    tl_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   100, 120]), np.array([10,  255, 255])),
        cv2.inRange(hsv, np.array([170, 100, 120]), np.array([180, 255, 255])),
    )
    tl_yellow = cv2.inRange(hsv, np.array([15, 100, 120]), np.array([35, 255, 255]))
    tl_green  = cv2.inRange(hsv, np.array([40,  60,  85]), np.array([85, 255, 255]))
    any_light = cv2.bitwise_or(cv2.bitwise_or(tl_red, tl_yellow), tl_green)

    k = np.ones((3, 3), np.uint8)
    any_light = cv2.morphologyEx(any_light, cv2.MORPH_OPEN,  k)
    any_light = cv2.morphologyEx(any_light, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(
        any_light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        # Blob must be small enough to be a single light bulb.
        # 0.006 × image_area ≈ a circle of radius ~4% of frame width —
        # large enough for a close-up bulb, too small to be foliage patches.
        if area < 20 or area > 0.006 * image_area:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # Traffic-light bulbs are nearly circular; tighten aspect ratio.
        # 0.5 – 2.0 allows for mild perspective distortion while rejecting
        # elongated leaf/branch blobs that the old 0.25-4.0 range accepted.
        aspect = w / float(h)
        if aspect < 0.50 or aspect > 2.0:
            continue

        # Brightness contrast: the blob must be brighter than its surroundings.
        # A painted sign reflects ambient light similarly to its background;
        # a glowing bulb stands out markedly.
        pad = max(w, h, 8)
        sx = max(0,            x - pad)
        sy = max(0,            y - pad)
        ex = min(image_width,  x + w + pad)
        ey = min(image_height, y + h + pad)

        blob_brightness     = float(np.mean(gray[y: y + h, x: x + w]))
        surround_brightness = float(np.mean(gray[sy:ey, sx:ex]))

        # Contrast threshold 12: overcast green bulbs can have only 12-18
        # units of contrast against the dark housing, yet they are still
        # self-emitting.  The surround_brightness < 130 gate below is the
        # primary false-positive guard; the contrast check just removes
        # blobs where the "surrounding" sample accidentally falls inside
        # the housing and is already close to blob brightness.
        if blob_brightness - surround_brightness < 12:
            continue

        # The surroundings must be dark (the housing).
        # A STOP sign sits in front of bright sky or a pole — its red face
        # is bright but its background is also bright, so surround ≥ 130.
        # A traffic-light bulb is embedded in a dark metal housing (gray ≈ 40-100).
        if surround_brightness >= 130:
            continue

        blobs.append({
            "x": x, "y": y, "w": w, "h": h,
            "cx": x + w // 2,
            "cy": y + h // 2,
            "area": area,
        })

    if not blobs:
        return []

    # Group blobs that are vertically aligned (same x, different y)
    blobs.sort(key=lambda b: b["cx"])
    used = [False] * len(blobs)
    detections = []

    for i, blob in enumerate(blobs):
        if used[i]:
            continue

        group = [blob]
        used[i] = True

        for j in range(i + 1, len(blobs)):
            if used[j]:
                continue
            other = blobs[j]

            # Must be horizontally close (within 1.5× the wider blob's width)
            x_tol = max(blob["w"], other["w"]) * 1.5
            if abs(blob["cx"] - other["cx"]) > x_tol:
                continue

            # Must be vertically separated but not too far apart
            y_dist = abs(blob["cy"] - other["cy"])
            max_gap = max(blob["h"], other["h"]) * 6
            if 3 < y_dist <= max_gap:
                group.append(other)
                used[j] = True

        # Bounding box of the group
        gx  = min(b["x"]          for b in group)
        gy  = min(b["y"]          for b in group)
        gex = max(b["x"] + b["w"] for b in group)
        gey = max(b["y"] + b["h"] for b in group)
        gw = gex - gx
        gh = gey - gy

        # --- Housing-aware bounding box ---
        # A standard traffic light has 3 lights arranged vertically.
        # Each "slot" = light diameter + ~30% spacing above/below.
        # We estimate how tall the full housing should be and extend the
        # box downward to cover any lights that weren't detected (e.g.
        # because they are off / too dim), so a single red blob still
        # produces a box that encompasses the whole 3-light column.
        avg_blob_w = sum(b["w"] for b in group) / len(group)
        avg_blob_h = sum(b["h"] for b in group) / len(group)

        slot_h     = avg_blob_h * 1.35          # one light + inter-slot gap
        target_h   = slot_h * 3                 # expected full 3-slot housing height
        extra_down = max(target_h - gh, 0)      # height missing below current group

        hpad   = max(int(avg_blob_w * 1.6), 8)
        vpad_t = max(int(avg_blob_h * 0.5), 4)
        # Cap vpad_b: the 3-slot formula can explode when a single large blob
        # is detected (e.g. warm foliage).  Never pad more than 40 % of the
        # image height downward.
        vpad_b = min(
            max(int(extra_down + avg_blob_h * 0.8), 8),
            int(image_height * 0.40),
        )

        bx = max(0,           gx - hpad)
        by = max(0,           gy - vpad_t)
        bw = min(image_width  - bx, gw + 2 * hpad)
        bh = min(image_height - by, gh + vpad_t + vpad_b)

        if bw < 8 or bh < 12:
            continue

        # Sanity check: a traffic-light box should never cover more than
        # 55 % of the frame in either dimension.  Boxes larger than this
        # are almost certainly false positives from warm foliage or signs.
        if bh > int(image_height * 0.55) or bw > int(image_width * 0.55):
            continue

        total_blob_area = sum(b["area"] for b in group)
        # Inflated score: wins NMS over any "Red" detection of the same bulb
        score = total_blob_area * 3 + 5000 * len(group)

        detections.append({
            "color": "Traffic Light",
            "bbox": (bx, by, bw, bh),
            "area": float(bw * bh),
            "fill_ratio": total_blob_area / float(bw * bh),
            "color_ratio": total_blob_area / float(bw * bh),
            "shape_name": "rectangle",
            "shape_sides": 4,
            "score": score,
        })

    return detections


def find_candidates(mask, image, color_name, min_area, max_area_ratio):
    """
    Find valid detections for one color mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for contour in contours:
        detection = build_detection(
            contour, mask, image, color_name, min_area, max_area_ratio
        )
        if detection is not None:
            detections.append(detection)

    return detections


def upscale_pass(image, scale=2.0):
    """
    Return detections found on a 2x upscaled image, scaled back to
    original coordinates.

    This is the primary fix for far/small signs: enlarging the image
    lets tiny blobs survive the minimum-area and minimum-size filters.
    Detections get a small score penalty (×0.85) so they don't beat a
    normal-scale detection of the same sign.
    """
    h, w = image.shape[:2]
    big = cv2.resize(image, (int(w * scale), int(h * scale)),
                     interpolation=cv2.INTER_LINEAR)

    hsv_big = convert_to_hsv(big)
    red_mask_big, blue_mask_big = create_color_masks(hsv_big, small=True)

    min_area_big = 60
    red_dets = find_candidates(red_mask_big, big, "Red",
                               min_area=min_area_big,
                               max_area_ratio=0.12)
    blue_dets = find_candidates(blue_mask_big, big, "Blue",
                                min_area=min_area_big * 2,
                                max_area_ratio=0.06)

    all_dets = red_dets + blue_dets

    # Scale bounding boxes back to original image coordinates
    for det in all_dets:
        x, y, w_b, h_b = det["bbox"]
        det["bbox"] = (
            int(x / scale),
            int(y / scale),
            max(1, int(w_b / scale)),
            max(1, int(h_b / scale)),
        )
        det["area"] = det["area"] / (scale * scale)
        det["score"] *= 0.85   # slight penalty vs normal-scale detections

    return all_dets


_LABEL_COLORS = {
    "Red": (0, 0, 255),
    "Blue": (255, 0, 0),
    "Traffic Light": (0, 165, 255),   # orange in BGR
}


def draw_and_extract(image, detections):
    """
    Draw bounding boxes and extract ROI images.
    """
    output = image.copy()
    rois = []

    for detection in detections:
        x, y, w, h = detection["bbox"]
        color = _LABEL_COLORS.get(detection["color"], (0, 255, 0))

        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            output,
            detection["color"],
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        roi = image[y: y + h, x: x + w]
        detection["roi"] = roi
        rois.append(roi)

    return output, rois


def detect_pipeline(image, min_area=80):
    """
    Run the full detection pipeline on one image.

    Changes from original:
    - min_area default lowered 120 -> 80 to catch smaller signs.
    - Added upscale_pass() so far/small signs are found too.
    - NMS now handles cross-color duplicates and containment suppression.
    - Hough circles are stricter (fewer ghost detections).
    - Shape classifier accepts 5-sided contours.
    - Color/fill thresholds relaxed for small/white-interior signs.
    - Morphological kernels reduced so tiny blobs are not erased.
    - Red saturation threshold raised to cut false positives.
    - Border/sky filters made size-aware so large signs are not dropped.
    - Traffic light detection added (dark housing + colored light blobs).
    """
    if image is None:
        raise ValueError("Input image is empty or not loaded correctly.")

    hsv = convert_to_hsv(image)
    red_mask, blue_mask = create_color_masks(hsv)

    # --- Normal-scale pass ---
    red_detections = find_candidates(
        red_mask,
        image,
        color_name="Red",
        min_area=max(40, int(min_area * 0.5)),
        max_area_ratio=0.30,  # 0.12→0.30: allow large close-up signs (stop sign)
    )

    blue_detections = find_candidates(
        blue_mask,
        image,
        color_name="Blue",
        min_area=min_area,
        max_area_ratio=0.25,  # 0.06→0.25: large close-up signs were rejected
    )

    red_circle_detections = detect_red_circles(image, red_mask)
    traffic_light_detections = detect_traffic_lights(image, hsv)

    # --- Upscale pass for far/small signs ---
    upscale_detections = upscale_pass(image, scale=2.0)

    all_detections = (
        red_detections
        + blue_detections
        + red_circle_detections
        + traffic_light_detections
        + upscale_detections
    )

    # Single NMS call that handles same-color AND cross-color duplicates.
    # Traffic Light detections have inflated scores so they survive over the
    # plain Red circle that may be detected inside the same housing.
    detections = remove_overlapping_detections(all_detections, iou_threshold=0.25)

    output, rois = draw_and_extract(image, detections)

    return {
        "output": output,
        "red_mask": red_mask,
        "blue_mask": blue_mask,
        "combined_mask": cv2.bitwise_or(red_mask, blue_mask),
        "rois": rois,
        "boxes": [d["bbox"] for d in detections],
        "detections": detections,
    }


def display_results(results, save_dir="out"):
    """
    Try to show images with OpenCV GUI; if not available (headless
    or headless OpenCV build), save the images to `save_dir` instead
    and print their paths for inspection.
    """
    try:
        cv2.imshow("Red Mask", results["red_mask"])
        cv2.imshow("Blue Mask", results["blue_mask"])
        cv2.imshow("Final Detection", results["output"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        os.makedirs(save_dir, exist_ok=True)
        red_path = os.path.join(save_dir, "red_mask.png")
        blue_path = os.path.join(save_dir, "blue_mask.png")
        out_path = os.path.join(save_dir, "final_detection.png")
        cv2.imwrite(red_path, results["red_mask"])
        cv2.imwrite(blue_path, results["blue_mask"])
        cv2.imwrite(out_path, results["output"])
        abs_dir = os.path.abspath(save_dir)
        print(f"GUI not available. Saved masks and output to: {abs_dir}")
        print(f" - {red_path}")
        print(f" - {blue_path}")
        print(f" - {out_path}")


if __name__ == "__main__":
    # Resolve image path relative to the project (script) directory so the
    # module can be executed from any working directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    rel_path = os.path.join("data", "processed", "gaussian", "road37.png")
    image_path = os.path.join(project_root, rel_path)

    # Fallback: if the path doesn't exist relative to the script, check the
    # current working directory (user may run the script from the project root).
    if not os.path.exists(image_path) and os.path.exists(rel_path):
        image_path = rel_path

    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Image not found at '{image_path}' (cwd={os.getcwd()})")

    results = detect_pipeline(image)

    print(f"Detected signs: {len(results['rois'])}")
    for detection in results["detections"]:
        print(
            f"{detection['color']} -> "
            f"box={detection['bbox']}, "
            f"area={detection['area']:.2f}, "
            f"color_ratio={detection['color_ratio']:.2f}"
        )

    display_results(results)