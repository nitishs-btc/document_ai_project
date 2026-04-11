import re
from collections import OrderedDict

from src.utils import KEY_LABELS, OPTION_LABELS, _estimate_line_threshold, is_section, merge_words, sort_words


def _cy(bbox):
    return (bbox[1] + bbox[3]) / 2


def _merge_text(items):
    text = ""

    for item in items:
        part = item["text"].strip()

        if not part or part == "|":
            continue

        if not text:
            text = part
            continue

        if part in {",", ".", ";", ":", "?", "!", "%", ")"}:
            text += part
            continue

        if text.endswith(("(", "/", "#", "$")):
            text += part
            continue

        text += f" {part}"

    return " ".join(text.split())


def _build_lines(items):
    ordered = sort_words(items)
    threshold = _estimate_line_threshold(ordered)
    lines = []

    for item in ordered:
        cy = _cy(item["bbox"])
        line = None

        for existing in lines:
            if abs(cy - existing["cy"]) <= threshold:
                line = existing
                break

        if line is None:
            line = {"cy": cy, "items": []}
            lines.append(line)

        line["items"].append(item)
        line["cy"] = sum(_cy(value["bbox"]) for value in line["items"]) / len(line["items"])

    normalized = []

    for idx, line in enumerate(sorted(lines, key=lambda value: value["cy"])):
        line_items = sorted(line["items"], key=lambda value: (value["bbox"][0], value["bbox"][1]))
        normalized.append({
            "index": idx,
            "cy": line["cy"],
            "items": line_items,
            "text": _merge_text(line_items),
        })

    return normalized


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for checkbox detection.") from exc

    return cv2


def _import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for checkbox classification.") from exc

    return np


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


def _vertical_overlap(box_a, box_b):
    return max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))


def _union_bbox(items):
    return [
        min(item["bbox"][0] for item in items),
        min(item["bbox"][1] for item in items),
        max(item["bbox"][2] for item in items),
        max(item["bbox"][3] for item in items),
    ]


def _normalize_option_text(text):
    normalized = " ".join(text.strip().split())
    letters_only = re.sub(r"[^a-z]", "", normalized.lower())

    if letters_only in {"yes", "ves"}:
        return "Yes"

    if letters_only in {"no", "ne"}:
        return "No"

    return normalized


def _is_directional_option_text(text):
    normalized = " ".join(text.strip().upper().split())
    return normalized in {"RIGHT-UP", "LEFT-UP", "RIGHT-DOWN", "LEFT-DOWN"}


def _looks_like_noise_option(text):
    letters_only = re.sub(r"[^a-z]", "", text.lower())

    if text.strip() in {"", "|", "-", "—", "0", "O"}:
        return True

    if letters_only in {"im", "aa", "yt", "daye", "hats"}:
        return True

    return len(letters_only) < 2 and text not in {"No"}


def _dedupe_options(options):
    deduped = []

    for option in sorted(options, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        option_text = option["text"].strip().lower()

        if not option_text:
            continue

        if any(
            existing["text"].strip().lower() == option_text
            and (
                _iou(existing["bbox"], option["bbox"]) > 0.45
                or abs(_cy(existing["bbox"]) - _cy(option["bbox"])) <= 18
            )
            for existing in deduped
        ):
            continue

        deduped.append(option)

    return deduped


def _collect_labeled_options(words_data):
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

    merged = []
    line_threshold = _estimate_line_threshold(option_words)
    merge_gap = max(18, line_threshold * 2)

    for word in option_words:
        if not merged:
            merged.append(word)
            continue

        prev = merged[-1]
        gap = word["bbox"][0] - prev["bbox"][2]

        if abs(_cy(word["bbox"]) - _cy(prev["bbox"])) <= line_threshold and -4 <= gap <= merge_gap:
            prev["text"] = _merge_text([prev, word])
            prev["bbox"] = [
                min(prev["bbox"][0], word["bbox"][0]),
                min(prev["bbox"][1], word["bbox"][1]),
                max(prev["bbox"][2], word["bbox"][2]),
                max(prev["bbox"][3], word["bbox"][3]),
            ]
        else:
            merged.append(word)

    return merged


def _cluster_anchor_positions(boxes, tolerance=52):
    anchors = []

    for box in sorted(boxes, key=lambda item: item[0]):
        if not anchors or abs(box[0] - anchors[-1]) > tolerance:
            anchors.append(box[0])
            continue

        anchors[-1] = int((anchors[-1] + box[0]) / 2)

    return anchors


def _recover_options_from_seed_boxes(gray, words_data, existing_options, seed_boxes):
    if not seed_boxes:
        return []

    merged_words = merge_words(words_data)
    lines = _build_lines(merged_words)
    line_threshold = _estimate_line_threshold(merged_words)
    anchors = _cluster_anchor_positions(seed_boxes)
    text_anchors = []

    for option in sorted(existing_options, key=lambda item: item["bbox"][0]):
        if "checkbox_bbox" not in option:
            continue

        x0 = option["bbox"][0]

        if not text_anchors or abs(x0 - text_anchors[-1]) > 50:
            text_anchors.append(x0)
            continue

        text_anchors[-1] = int((text_anchors[-1] + x0) / 2)

    y_padding = max(40, line_threshold * 2)
    min_y = min(box[1] for box in seed_boxes) - y_padding
    max_y = max(box[3] for box in seed_boxes) + y_padding
    recovered = []

    for box in _detect_checkboxes_globally(gray):
        if any(_iou(box, seed_box) > 0.45 for seed_box in seed_boxes):
            continue

        if box[1] < min_y or box[3] > max_y:
            continue

        if not any(abs(box[0] - anchor) <= 48 for anchor in anchors):
            continue

        line_index = _find_line_index(lines, _cy(box))

        if line_index is None:
            continue

        line = lines[line_index]

        if abs(line["cy"] - _cy(box)) > max(34, line_threshold * 1.8):
            continue

        candidates = [
            item for item in line["items"]
            if item["bbox"][0] >= box[2] - 10
            and _vertical_overlap(item["bbox"], box) > 0
            and not is_section(item)
        ]

        if not candidates:
            continue

        candidates = sorted(candidates, key=lambda item: (item["bbox"][0], item["bbox"][1]))
        first_gap = candidates[0]["bbox"][0] - box[2]

        if first_gap > 220:
            continue

        cluster = [candidates[0]]
        max_gap = max(100, line_threshold * 5)

        for item in candidates[1:]:
            gap = item["bbox"][0] - cluster[-1]["bbox"][2]

            if gap > max_gap:
                break

            cluster.append(item)

        text = _normalize_option_text(_merge_text(cluster))

        if (
            not _is_valid_option_text(text)
            or _is_directional_option_text(text)
            or _looks_like_noise_option(text)
            or len(text) > 64
            or len(text.split()) > 6
        ):
            continue

        option_bbox = _union_bbox(cluster)

        if text_anchors and not any(abs(option_bbox[0] - anchor) <= 50 for anchor in text_anchors):
            continue

        if any(_iou(option_bbox, option["bbox"]) > 0.55 for option in existing_options + recovered):
            continue

        recovered.append({
            "text": text,
            "bbox": option_bbox,
            "label": "O",
            "checkbox_bbox": box,
        })

    return recovered


def _detect_box_for_option(gray, option):
    cv2 = _import_cv2()
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
    cv2 = _import_cv2()
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
    cv2 = _import_cv2()
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if words_data:
        options = _collect_option_candidates(words_data, img_path=img_path)

        if not options:
            return []

        boxes = []

        for option in options:
            box = option.get("checkbox_bbox") or _detect_box_for_option(gray, option)

            if box:
                boxes.append(box)

        return _dedupe_boxes(boxes)

    return []


def classify_checkbox(img_path, boxes):
    cv2 = _import_cv2()
    np = _import_numpy()
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


def _collect_option_candidates(words_data, img_path=None):
    options = _collect_labeled_options(words_data)

    if not img_path or not options:
        return _dedupe_options(options)

    cv2 = _import_cv2()
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    seeded_options = []
    seed_boxes = []

    for option in options:
        option_copy = dict(option)
        box = _detect_box_for_option(gray, option_copy)

        if box:
            option_copy["checkbox_bbox"] = box
            seed_boxes.append(box)

        seeded_options.append(option_copy)

    recovered = _recover_options_from_seed_boxes(gray, words_data, seeded_options, seed_boxes)
    return _dedupe_options(seeded_options + recovered)


def _find_line_index(lines, cy):
    best = None

    for line in lines:
        score = abs(line["cy"] - cy)

        if best is None or score < best[0]:
            best = (score, line["index"])

    return best[1] if best else None


def _current_section_for_line(lines, line_index):
    for idx in range(line_index, -1, -1):
        if any(is_section(item) for item in lines[idx]["items"]):
            return _merge_text(lines[idx]["items"]).strip()

    return "GENERAL"


def _find_option_for_checkbox(checkbox, options, used_option_indices):
    x0, y0, x1, y1 = checkbox["bbox"]
    cy = (y0 + y1) / 2
    best = None

    for idx, option in enumerate(options):
        if idx in used_option_indices:
            continue

        option_box = option.get("checkbox_bbox")

        if option_box:
            overlap = _iou(checkbox["bbox"], option_box)

            if overlap > 0.45:
                score = -200 + (1 - overlap)

                if best is None or score < best[0]:
                    best = (score, idx, option)

                continue

        wx0, wy0, wx1, wy1 = option["bbox"]
        wcy = _cy(option["bbox"])
        vertical_delta = abs(wcy - cy)

        if vertical_delta > max(34, (wy1 - wy0) * 1.6):
            continue

        if wx0 >= x1 - 15:
            horizontal_delta = wx0 - x1

            if horizontal_delta > 340:
                continue

            score = (vertical_delta * 4) + max(0, horizontal_delta)
        else:
            horizontal_delta = x0 - wx1

            if horizontal_delta < -30 or horizontal_delta > 120:
                continue

            score = (vertical_delta * 5) + max(0, horizontal_delta) + 90

        if best is None or score < best[0]:
            best = (score, idx, option)

    return best


def _question_from_option_line(line, first_option_x0):
    prompt_items = [
        item for item in line["items"]
        if item["bbox"][2] <= first_option_x0 - 20 and not is_section(item)
    ]

    if not prompt_items:
        return ""

    prompt = _merge_text(prompt_items).strip()

    if _looks_like_noise_option(prompt):
        return ""

    return prompt


def _question_from_previous_lines(lines, line_index):
    section_name = ""

    for idx in range(line_index - 1, -1, -1):
        line = lines[idx]

        if any(is_section(item) for item in line["items"]):
            section_name = _merge_text(line["items"]).strip()

            if section_name:
                break

            continue

        if any(item.get("label") in OPTION_LABELS for item in line["items"]) and not any(
            item.get("label") in KEY_LABELS for item in line["items"]
        ):
            continue

        key_items = [
            item for item in line["items"]
            if item.get("label") in KEY_LABELS or item["text"].strip().endswith((':', '?'))
        ]

        if key_items:
            key_text = _merge_text(key_items).strip()
            line_text = line["text"].strip()

            if key_text.endswith(":") or key_text.endswith("?"):
                if line_text.endswith(":") or line_text.endswith("?"):
                    return line_text

                return key_text

        text = line["text"].strip()

        if text.endswith(":") or text.endswith("?"):
            return text

    return section_name


def _option_prompt(lines, option_line_index, options):
    line = lines[option_line_index]
    line_option_x0s = [
        option["bbox"][0]
        for option in options
        if _find_line_index(lines, _cy(option["bbox"])) == option_line_index
    ]

    if line_option_x0s:
        question = _question_from_option_line(line, min(line_option_x0s))

        if question:
            return question

    return _question_from_previous_lines(lines, option_line_index)


def map_checkbox_to_text(checkboxes, words_data, img_path=None):
    option_blocks = _collect_option_candidates(words_data, img_path=img_path)

    if not option_blocks:
        return []

    merged_words = merge_words(words_data)
    lines = _build_lines(merged_words)
    used_option_indices = set()
    results = []

    for cb in sorted(checkboxes, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        best_option = _find_option_for_checkbox(cb, option_blocks, used_option_indices)

        if not best_option:
            continue

        _, option_idx, option = best_option
        used_option_indices.add(option_idx)
        line_index = _find_line_index(lines, _cy(option["bbox"]))
        question = ""
        section = "GENERAL"

        if line_index is not None:
            question = _option_prompt(lines, line_index, option_blocks)
            section = _current_section_for_line(lines, line_index)

        results.append({
            "text": option["text"].strip(),
            "option": option["text"].strip(),
            "question": question,
            "section": section,
            "status": cb["status"]
        })

    return results


def group_checkbox_results(checkboxes):
    grouped = OrderedDict()

    for item in checkboxes:
        section = item.get("section") or "GENERAL"
        question = item.get("question") or item.get("option") or "options"
        grouped.setdefault(section, OrderedDict())

        if question not in grouped[section]:
            grouped[section][question] = OrderedDict()

        grouped[section][question][item["option"]] = item["status"]

    return grouped
