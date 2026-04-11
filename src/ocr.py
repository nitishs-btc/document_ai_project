import json
import os

import pytesseract

DEBUG_OCR_DIR = "debug/ocr"
os.makedirs(DEBUG_OCR_DIR, exist_ok=True)

OCR_CONFIGS = (
    ("--oem 3 --psm 4", 40),
    ("--oem 3 --psm 11", 30),
)


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


def _normalize_text(text):
    return " ".join(text.strip().lower().split())


def _dedupe_candidates(candidates):
    kept = []

    for candidate in sorted(
        candidates,
        key=lambda item: (item["conf"], (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1])),
        reverse=True,
    ):
        duplicate = False

        for existing in kept:
            overlap = _iou(candidate["bbox"], existing["bbox"])

            if overlap > 0.8:
                duplicate = True
                break

            if overlap > 0.5 and _normalize_text(candidate["text"]) == _normalize_text(existing["text"]):
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    return sorted(kept, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def run_ocr(image_path):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for OCR processing.") from exc

    img = cv2.imread(image_path)
    candidates = []

    for custom_config, confidence_floor in OCR_CONFIGS:
        data = pytesseract.image_to_data(
            img,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )

        for i in range(len(data["text"])):
            text = data["text"][i].strip()

            if not text:
                continue

            raw_conf = data["conf"][i]
            conf = float(raw_conf) if raw_conf != "-1" else -1.0

            if conf < confidence_floor:
                continue

            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            candidates.append({
                "text": text,
                "bbox": [x, y, x + w, y + h],
                "conf": conf,
            })

    deduped = _dedupe_candidates(candidates)
    words_data = [{"text": item["text"], "bbox": item["bbox"]} for item in deduped]
    debug_output = [
        {"text": item["text"], "bbox": item["bbox"], "conf": item["conf"]}
        for item in deduped
    ]

    filename = os.path.basename(image_path).replace(".png", ".json")

    with open(os.path.join(DEBUG_OCR_DIR, filename), "w") as handle:
        json.dump(debug_output, handle, indent=2)

    return words_data
