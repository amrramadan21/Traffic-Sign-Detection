import cv2
import numpy as np


def convert_to_hsv(image):
    """
    Convert the input image from BGR to HSV.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def clean_mask(mask, color_name):
    """
    Remove small noise from a mask.
    Blue uses stronger cleanup to reduce sky detections.
    """
    if color_name == "Blue":
        open_kernel = np.ones((5, 5), np.uint8)
        close_kernel = np.ones((7, 7), np.uint8)
    else:
        open_kernel = np.ones((3, 3), np.uint8)
        close_kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask


def create_color_masks(hsv):
    """
    Build red and blue masks in HSV space.
    Red uses two ranges because hue wraps around in HSV.
    """
    lower_red1 = np.array([0, 45, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 45, 40])
    upper_red2 = np.array([180, 255, 255])

    lower_blue = np.array([100, 120, 50])
    upper_blue = np.array([128, 255, 255])

    red_mask_1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask_2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    red_mask = clean_mask(red_mask, "Red")
    blue_mask = clean_mask(blue_mask, "Blue")

    return red_mask, blue_mask


def classify_shape(contour):
    """
    Keep simple sign-like shapes: triangle, rectangle, circle-like.
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
    if sides >= 6:
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
    Compute overlap between two boxes.
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


def remove_overlapping_detections(detections, overlap_threshold=0.3):
    """
    Keep the strongest detection when boxes overlap.
    """
    detections = sorted(detections, key=lambda item: item["score"], reverse=True)
    filtered = []

    for detection in detections:
        keep_detection = True

        for saved_detection in filtered:
            same_color = detection["color"] == saved_detection["color"]
            overlap = calculate_iou(detection["bbox"], saved_detection["bbox"])

            if same_color and overlap > overlap_threshold:
                keep_detection = False
                break

        if keep_detection:
            filtered.append(detection)

    return filtered


def build_detection(contour, mask, image, color_name, min_area, max_area_ratio):
    """
    Convert one contour into a detection if it passes all filters.
    """
    image_height, image_width = image.shape[:2]
    image_area = image_height * image_width
    area = cv2.contourArea(contour)

    if area < min_area or area > max_area_ratio * image_area:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    if w < 12 or h < 12:
        return None

    if color_name == "Blue" and touches_border(x, y, w, h, image_width, image_height):
        return None

    # Large blue regions near the top are often sky.
    if color_name == "Blue" and y < int(0.25 * image_height) and area > 0.01 * image_area:
        return None

    aspect_ratio = w / float(h)
    if color_name == "Blue":
        if aspect_ratio < 0.60 or aspect_ratio > 1.40:
            return None
    else:
        if aspect_ratio < 0.45 or aspect_ratio > 1.60:
            return None

    bbox_area = float(w * h)
    fill_ratio = area / bbox_area
    min_fill_ratio = 0.25 if color_name == "Blue" else 0.10
    if fill_ratio < min_fill_ratio:
        return None

    roi_mask = mask[y : y + h, x : x + w]
    color_pixels = cv2.countNonZero(roi_mask)
    color_ratio = color_pixels / bbox_area
    min_color_ratio = 0.30 if color_name == "Blue" else 0.08
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
    Recover circular red signs with thin red borders.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=100,
        param2=18,
        minRadius=8,
        maxRadius=80,
    )

    if circles is None:
        return []

    detections = []
    image_height, image_width = image.shape[:2]

    for center_x, center_y, radius in np.round(circles[0]).astype(int):
        x = max(center_x - radius, 0)
        y = max(center_y - radius, 0)
        w = min(2 * radius, image_width - x)
        h = min(2 * radius, image_height - y)

        if w < 12 or h < 12:
            continue

        ring_mask = np.zeros_like(red_mask)
        cv2.circle(ring_mask, (center_x, center_y), radius, 255, 2)
        overlap_mask = cv2.bitwise_and(red_mask, ring_mask)

        ring_pixels = cv2.countNonZero(ring_mask)
        red_pixels = cv2.countNonZero(overlap_mask)
        color_ratio = red_pixels / float(ring_pixels) if ring_pixels else 0.0

        if color_ratio < 0.08:
            continue

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


def find_candidates(mask, image, color_name, min_area, max_area_ratio):
    """
    Find valid detections for one color mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for contour in contours:
        detection = build_detection(contour, mask, image, color_name, min_area, max_area_ratio)
        if detection is not None:
            detections.append(detection)

    return detections


def draw_and_extract(image, detections):
    """
    Draw bounding boxes and extract ROI images.
    """
    output = image.copy()
    rois = []

    for detection in detections:
        x, y, w, h = detection["bbox"]
        color = (0, 0, 255) if detection["color"] == "Red" else (255, 0, 0)

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

        roi = image[y : y + h, x : x + w]
        detection["roi"] = roi
        rois.append(roi)

    return output, rois


def detect_pipeline(image, min_area=120):
    """
    Run the full detection pipeline on one image.
    """
    if image is None:
        raise ValueError("Input image is empty or not loaded correctly.")

    hsv = convert_to_hsv(image)
    red_mask, blue_mask = create_color_masks(hsv)

    red_detections = find_candidates(
        red_mask,
        image,
        color_name="Red",
        min_area=max(60, int(min_area * 0.6)),
        max_area_ratio=0.12,
    )

    blue_detections = find_candidates(
        blue_mask,
        image,
        color_name="Blue",
        min_area=min_area,
        max_area_ratio=0.06,
    )

    red_circle_detections = detect_red_circles(image, red_mask)

    detections = remove_overlapping_detections(
        red_detections + blue_detections + red_circle_detections
    )

    output, rois = draw_and_extract(image, detections)

    return {
        "output": output,
        "red_mask": red_mask,
        "blue_mask": blue_mask,
        "combined_mask": cv2.bitwise_or(red_mask, blue_mask),
        "rois": rois,
        "boxes": [detection["bbox"] for detection in detections],
        "detections": detections,
    }


if __name__ == "__main__":
    image_path = "data/processed/gaussian/road183.png"
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found!")

    results = detect_pipeline(image)

    print(f"Detected signs: {len(results['rois'])}")

    for detection in results["detections"]:
        print(
            f"{detection['color']} -> "
            f"box={detection['bbox']}, "
            f"area={detection['area']:.2f}, "
            f"color_ratio={detection['color_ratio']:.2f}"
        )

    cv2.imshow("Red Mask", results["red_mask"])
    cv2.imshow("Blue Mask", results["blue_mask"])
    cv2.imshow("Final Detection", results["output"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()