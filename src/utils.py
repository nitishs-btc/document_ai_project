from collections import OrderedDict
from statistics import median

KEY_LABELS = {"B-KEY", "I-KEY"}
VALUE_LABELS = {"B-VALUE", "I-VALUE"}
SECTION_LABELS = {"B-SECTION", "I-SECTION"}
OPTION_LABELS = {"B-OPTION", "I-OPTION"}


def _cx(bbox):
    return (bbox[0] + bbox[2]) / 2


def _cy(bbox):
    return (bbox[1] + bbox[3]) / 2


def _width(bbox):
    return max(1, bbox[2] - bbox[0])


def _height(bbox):
    return max(1, bbox[3] - bbox[1])


def _estimate_line_threshold(items):
    heights = [_height(item["bbox"]) for item in items if item.get("bbox")]

    if not heights:
        return 14

    return max(12, int(median(heights) * 0.7))


def _is_same_line(a, b, threshold=None):
    if threshold is None:
        threshold = 16

    return abs(_cy(a) - _cy(b)) <= threshold


# =========================
# STABLE WORD ORDERING
# =========================

def sort_words(words):
    if not words:
        return []

    threshold = _estimate_line_threshold(words)
    ordered = sorted(words, key=lambda w: (_cy(w["bbox"]), w["bbox"][0]))
    lines = []

    for word in ordered:
        cy = _cy(word["bbox"])
        line = None

        for existing in lines:
            if abs(cy - existing["cy"]) <= threshold:
                line = existing
                break

        if line is None:
            line = {"cy": cy, "words": []}
            lines.append(line)

        line["words"].append(word)
        line["cy"] = sum(_cy(item["bbox"]) for item in line["words"]) / len(line["words"])

    sorted_words = []

    for line in sorted(lines, key=lambda item: item["cy"]):
        line["words"].sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
        sorted_words.extend(line["words"])

    return sorted_words


# =========================
# TEXT FILTER
# =========================

def _is_valid_text(text):
    t = text.strip().lower()

    if len(t) < 2:
        return False

    if t in ["|", "-", "—", ".", ","]:
        return False

    if not any(c.isalpha() for c in t):
        return False

    return True


def _is_noise_value(text):
    return text.strip() in {"", "|", "/", "-", "—", "_"}


def _looks_like_key(item):
    text = item["text"].strip()
    label = item.get("label", "O")

    if label in KEY_LABELS:
        return True

    return label == "O" and text.endswith(":") and _is_valid_text(text[:-1])


def _looks_like_value(item):
    text = item["text"].strip()
    label = item.get("label", "O")

    if label in VALUE_LABELS:
        return True

    if label != "O" or _looks_like_key(item) or is_section(item) or _is_noise_value(text):
        return False

    if text.isupper() and len(text.split()) > 2:
        return False

    return True


# =========================
# CLEAN KEY
# =========================

def clean_key(text):
    cleaned = " ".join(text.strip().split())
    lowered = cleaned.lower()

    bad_tokens = {"ph", "one", "im", "dp", "fa", "|"}

    if lowered in bad_tokens:
        return None

    if len(lowered) < 3:
        return None

    if any(token in lowered for token in ["lll", "|||"]):
        return None

    return cleaned


# =========================
# SECTION DETECTION
# =========================

def is_section(item):
    text = item["text"].strip()
    label = item.get("label", "O")

    if not text:
        return False

    if label in SECTION_LABELS:
        return True

    if not _is_valid_text(text):
        return False

    words = text.replace("/", " ").split()
    return text.isupper() and 2 <= len(words) <= 6 and _width(item["bbox"]) >= 150


# =========================
# MERGE WORDS
# =========================

def merge_words(words_data):
    ordered_words = sort_words(words_data)
    merged = []
    line_threshold = _estimate_line_threshold(ordered_words)
    merge_gap = max(24, line_threshold * 3)

    for word in ordered_words:
        text = word["text"].strip()

        if not text:
            continue

        if text == "|":
            continue

        bbox = word["bbox"]
        label = word.get("label", "O")

        if not merged:
            merged.append({"text": text, "bbox": bbox.copy(), "label": label})
            continue

        prev = merged[-1]
        gap = bbox[0] - prev["bbox"][2]

        if (
            label == prev["label"]
            and (label != "O" or (":" not in prev["text"] and ":" not in text))
            and _is_same_line(prev["bbox"], bbox, line_threshold)
            and -5 <= gap <= merge_gap
        ):
            prev["text"] = f"{prev['text']} {text}".strip()
            prev["bbox"] = [
                min(prev["bbox"][0], bbox[0]),
                min(prev["bbox"][1], bbox[1]),
                max(prev["bbox"][2], bbox[2]),
                max(prev["bbox"][3], bbox[3]),
            ]
        else:
            merged.append({"text": text, "bbox": bbox.copy(), "label": label})

    return merged


# =========================
# KV PAIRING
# =========================

def _score_value_for_key(key_item, value_item, line_threshold):
    key_bbox = key_item["bbox"]
    value_bbox = value_item["bbox"]
    value_text = value_item["text"].strip()

    if _is_noise_value(value_text):
        return None

    cy_diff = abs(_cy(key_bbox) - _cy(value_bbox))

    if cy_diff <= line_threshold and value_bbox[0] >= key_bbox[2] - 5:
        gap = max(0, value_bbox[0] - key_bbox[2])
        score = gap + (cy_diff * 4)

        if len(value_text) <= 2:
            score += 40

        if value_item.get("label") == "O":
            score += 25

        return score

    if value_bbox[1] < key_bbox[3] - line_threshold:
        return None

    vertical_gap = max(0, value_bbox[1] - key_bbox[3])

    if vertical_gap > max(110, line_threshold * 6):
        return None

    overlap = max(0, min(key_bbox[2], value_bbox[2]) - max(key_bbox[0], value_bbox[0]))
    center_dx = abs(_cx(key_bbox) - _cx(value_bbox))

    if overlap == 0 and center_dx > max(240, _width(key_bbox) * 1.4):
        return None

    score = 80 + (vertical_gap * 3) + (center_dx * 0.35) - (overlap * 0.2)

    if value_item.get("label") == "O":
        score += 25

    return score


def extract_key_value_pairs(merged):
    pairings = {}
    used_value_indices = set()
    line_threshold = _estimate_line_threshold(merged)

    key_items = [
        (idx, item)
        for idx, item in enumerate(merged)
        if _looks_like_key(item)
    ]
    value_items = [
        (idx, item)
        for idx, item in enumerate(merged)
        if _looks_like_value(item)
    ]

    for key_idx, key_item in key_items:
        best_match = None

        for value_idx, value_item in value_items:
            if value_idx in used_value_indices:
                continue

            score = _score_value_for_key(key_item, value_item, line_threshold)

            if score is None:
                continue

            if best_match is None or score < best_match[0]:
                best_match = (score, value_idx, value_item["text"].strip())

        if best_match is None:
            pairings[key_idx] = ""
            continue

        _, value_idx, value_text = best_match
        used_value_indices.add(value_idx)
        pairings[key_idx] = value_text

    return pairings


# =========================
# STRUCTURED OUTPUT
# =========================

def extract_structured(words_data, labels=None):
    if labels:
        for word, label in zip(words_data, labels):
            word["label"] = label

    merged = merge_words(words_data)

    for idx, item in enumerate(merged):
        item["_idx"] = idx

    key_value_pairs = extract_key_value_pairs(merged)
    structured = OrderedDict()
    current_section = None

    for item in merged:
        text = item["text"].strip()

        if is_section(item):
            current_section = text
            structured.setdefault(current_section, OrderedDict())
            continue

        if not _looks_like_key(item):
            continue

        key = clean_key(text)

        if not key:
            continue

        if not current_section:
            current_section = "GENERAL"
            structured.setdefault(current_section, OrderedDict())

        structured[current_section][key] = key_value_pairs.get(item["_idx"], "")

    return OrderedDict((section, values) for section, values in structured.items() if values)
