import cv2
import numpy as np


def detect_checkboxes(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # ✅ Strict checkbox filter
        if 40 < w < 80 and 40 < h < 80:
            aspect_ratio = w / float(h)

            if 0.95 < aspect_ratio < 1.05:  # square
                boxes.append((x, y, w, h))

    return boxes


def classify_checkbox(img_path, boxes):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    results = []

    for (x, y, w, h) in boxes:
        roi = gray[y:y + h, x:x + w]

        # Count dark pixels
        _, roi_thresh = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY_INV)
        filled_ratio = np.sum(roi_thresh > 0) / (w * h)

        if filled_ratio > 0.2:
            status = "checked"
        else:
            status = "unchecked"

        results.append({
            "bbox": [x, y, x + w, y + h],
            "status": status
        })

    return results


def map_checkbox_to_text(checkboxes, words_data):
    results = []

    for cb in checkboxes:
        x0, y0, x1, y1 = cb["bbox"]
        cy = (y0 + y1) / 2

        label_words = []

        for word in words_data:
            wx0, wy0, wx1, wy1 = word["bbox"]
            wcy = (wy0 + wy1) / 2

            # Same row
            if abs(wcy - cy) < 20 and wx0 > x1:
                label_words.append(word["text"])

        label = " ".join(label_words).strip()

        if len(label) > 2:
            results.append({
                "label": label,
                "status": cb["status"]
            })

    return results