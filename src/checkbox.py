import cv2
import numpy as np


def detect_checkboxes(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 15 or h < 15 or w > 60 or h > 60:
            continue

        if not (0.95 < w / float(h) < 1.05):
            continue

        boxes.append((x, y, w, h))

    return boxes


def classify_checkbox(img_path, boxes):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    results = []

    for (x, y, w, h) in boxes:
        roi = gray[y:y+h, x:x+w]

        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        status = "checked" if edge_density > 0.08 else "unchecked"

        results.append({
            "bbox": [x, y, x+w, y+h],
            "status": status
        })

    return results


def map_checkbox_to_text(checkboxes, words_data):
    results = []

    valid_words = [
        w for w in words_data
        if w.get("label") in ["B-OPTION", "I-OPTION"]
        and len(w["text"]) > 2
    ]

    for cb in checkboxes:
        x0, y0, x1, y1 = cb["bbox"]
        cy = (y0 + y1) / 2

        best = None
        best_dist = 9999

        for w in valid_words:
            wx0, wy0, wx1, wy1 = w["bbox"]
            wcy = (wy0 + wy1) / 2

            if abs(wcy - cy) < 25:
                dist = abs(wx0 - x1)

                if dist < best_dist and dist < 120:
                    best_dist = dist
                    best = w["text"]

        if best:
            results.append({
                "text": best,
                "status": cb["status"]
            })

    return results