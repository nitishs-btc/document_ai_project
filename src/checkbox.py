import cv2
import numpy as np

from src.utils import OPTION_LABELS, merge_words, sort_words


def _cy(bbox):
    return (bbox[1] + bbox[3]) / 2


def _iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)

    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0

    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)

    return inter_area / float(area_a + area_b - inter_area)


def _dedupe_boxes(boxes):
    deduped = []

    for box in sorted(boxes, key=lambda item: ((item[2] - item[0]) * (item[3] - item[1])), reverse=True):
        if any(_iou(box, kept) > 0.5 for kept in deduped):
            continue

        deduped.append(box)

    return sorted(deduped, key=lambda item: (item[1], item[0]))


def _is_valid_option_text(text):
    stripped = text.strip()

    if stripped in {"", "|", "-", "—"}:
        return False

    return any(char.isalnum() for char in stripped)


def _collect_options(words_data):
    option_words = []

    for word in sort_words(words_data):
        text = word["text"].strip()

        if word.get("label") not in OPTION_LABELS:
            continue

        if text == "|" or not _is_valid_option_text(text):
            continue

        option_words.append({
            "text": text,
            "bbox": word["bbox"].copy(),
            "label": word["label"],
        })

    return merge_words(option_words)


def _detect_box_for_option(gray, option):
    x0, y0, x1, y1 = option["bbox"]
    search_x0 = max(0, x0 - 180)
    search_x1 = max(search_x0 + 1, x0 - 5)
    search_y0 = max(0, y0 - 20)
    search_y1 = min(gray.shape[0], y1 + 20)

    roi = gray[search_y0:search_y1, search_x0:search_x1]

    if roi.size == 0:
        return None

    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    best_box = None
    target_cy = _cy(option["bbox"])

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 18 or h < 18 or w > 60 or h > 60:
            continue

        if not (0.75 <= w / float(h) <= 1.25):
            continue

        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

        if not (4 <= len(approx) <= 8):
            continue

        box = [search_x0 + x, search_y0 + y, search_x0 + x + w, search_y0 + y + h]
        box_cy = _cy(box)
        score = (abs(box_cy - target_cy) * 4) + abs(box[2] - x0)

        if best_box is None or score < best_box[0]:
            best_box = (score, box)

    if best_box is None:
        return None

    return best_box[1]


def _detect_checkboxes_globally(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if w < 24 or h < 24 or w > 60 or h > 60:
            continue

        if not (0.8 <= w / float(h) <= 1.2):
            continue

        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

        if not (4 <= len(approx) <= 8):
            continue

        boxes.append([x, y, x + w, y + h])

    return _dedupe_boxes(boxes)


def detect_checkboxes(img_path, words_data=None):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if words_data:
        boxes = []

        for option in _collect_options(words_data):
            box = _detect_box_for_option(gray, option)

            if box:
                boxes.append(box)

        if boxes:
            return _dedupe_boxes(boxes)

    return _detect_checkboxes_globally(gray)


def classify_checkbox(img_path, boxes):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8
    )

    results = []

    for (x0, y0, x1, y1) in boxes:
        w = x1 - x0
        h = y1 - y0
        pad = max(2, int(min(w, h) * 0.2))

        inner_x0 = min(max(x0 + pad, 0), x1)
        inner_y0 = min(max(y0 + pad, 0), y1)
        inner_x1 = max(min(x1 - pad, gray.shape[1]), inner_x0)
        inner_y1 = max(min(y1 - pad, gray.shape[0]), inner_y0)

        roi_binary = binary[inner_y0:inner_y1, inner_x0:inner_x1]
        roi_hsv = hsv[inner_y0:inner_y1, inner_x0:inner_x1]

        if roi_binary.size == 0:
            continue

        ink_ratio = float(np.count_nonzero(roi_binary)) / float(roi_binary.size)

        hue = roi_hsv[:, :, 0]
        sat = roi_hsv[:, :, 1]
        val = roi_hsv[:, :, 2]
        red_mask = ((hue <= 12) | (hue >= 170)) & (sat >= 70) & (val >= 40)
        red_ratio = float(np.count_nonzero(red_mask)) / float(red_mask.size)

        status = "checked" if red_ratio > 0.01 or ink_ratio > 0.12 else "unchecked"

        results.append({
            "bbox": [x0, y0, x1, y1],
            "status": status
        })

    return results


def map_checkbox_to_text(checkboxes, words_data):
    merged_words = _collect_options(words_data)
    results = []
    used_option_indices = set()

    valid_words = [
        word for word in merged_words
        if word.get("label") in OPTION_LABELS
        and _is_valid_option_text(word["text"])
    ]

    for cb in sorted(checkboxes, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        x0, y0, x1, y1 = cb["bbox"]
        cy = (y0 + y1) / 2

        best = None

        for idx, word in enumerate(valid_words):
            if idx in used_option_indices:
                continue

            wx0, wy0, wx1, wy1 = word["bbox"]
            wcy = _cy(word["bbox"])

            if abs(wcy - cy) > max(28, (wy1 - wy0) * 1.4):
                continue

            horizontal_gap = wx0 - x1

            if not (-10 <= horizontal_gap <= 280):
                continue

            score = (abs(wcy - cy) * 3) + max(0, horizontal_gap)

            if best is None or score < best[0]:
                best = (score, idx, word["text"].strip())

        if best:
            used_option_indices.add(best[1])
            results.append({
                "text": best[2],
                "status": cb["status"]
            })

    return results
