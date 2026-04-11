import re
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


def _union_bbox(items):
    return [
        min(item["bbox"][0] for item in items),
        min(item["bbox"][1] for item in items),
        max(item["bbox"][2] for item in items),
        max(item["bbox"][3] for item in items),
    ]


def _estimate_line_threshold(items):
    heights = [_height(item["bbox"]) for item in items if item.get("bbox")]

    if not heights:
        return 14

    return max(12, int(median(heights) * 0.7))


def _is_same_line(a, b, threshold=None):
    if threshold is None:
        threshold = 16

    return abs(_cy(a) - _cy(b)) <= threshold


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

    normalized_lines = []

    for idx, line in enumerate(sorted(lines, key=lambda value: value["cy"])):
        line_items = sorted(line["items"], key=lambda value: (value["bbox"][0], value["bbox"][1]))
        normalized_lines.append({
            "index": idx,
            "cy": line["cy"],
            "items": line_items,
            "bbox": _union_bbox(line_items),
        })

    return normalized_lines


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


def _looks_like_key_connector(item):
    text = item["text"].strip().lower()
    label = item.get("label", "O")

    if label != "O":
        return False

    if text in {"of", "and", "for", "to", "or", "/", "-", "&", "(", ")"}:
        return True

    return text.isdigit() and len(text) <= 3


# =========================
# CLEAN KEY
# =========================

def clean_key(text):
    cleaned = " ".join(text.strip().split())
    cleaned = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cleaned)
    cleaned = re.sub(r"\?0\b", "?", cleaned)
    cleaned = re.sub(r"\bPh\s+one\b", "Phone", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bDo\s*B\b", "DOB", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([:;,?.!])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    cleaned = re.sub(r"\s+/\s+", " / ", cleaned)
    cleaned = re.sub(r"\s*:\s*$", ":", cleaned)
    cleaned = " ".join(cleaned.split())

    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1].strip()

    lowered = cleaned.lower()

    bad_tokens = {"ph", "one", "im", "dp", "fa", "|"}

    if lowered in bad_tokens:
        return None

    if len(lowered) < 3:
        return None

    if any(token in lowered for token in ["lll", "|||"]):
        return None

    return cleaned


def clean_section(text):
    cleaned = " ".join(text.strip().split())
    cleaned = re.sub(r"\s+([:;,?.!])", r"\1", cleaned)
    cleaned = re.sub(r"\s+/\s+", " / ", cleaned)
    cleaned = re.sub(r"\s*-\s*", " - ", cleaned)
    cleaned = " ".join(cleaned.split()).strip(" -")
    return cleaned


def clean_value(text):
    cleaned = " ".join(text.strip().split())
    cleaned = re.sub(r"\s+([,:;?.!])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    cleaned = re.sub(r"\s+/\s+", " / ", cleaned)
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

    if ":" in text or any(char.isdigit() for char in text):
        return False

    if text.startswith("(") or text.count("(") > 1:
        return False

    if text.endswith(")") or "/" in text:
        return False

    words = text.replace("/", " ").split()
    return text.isupper() and 2 <= len(words) <= 6 and _width(item["bbox"]) >= 150


def _extract_section_from_line(line):
    section_items = []

    for item in line["items"]:
        text = item["text"].strip()

        if is_section(item):
            section_items.append(item)
            continue

        if section_items and item.get("label", "O") == "O" and (text in {"-", "/", "&"} or text.isdigit()):
            section_items.append(item)

    if not section_items:
        return None

    return clean_section(_merge_text(section_items))


def _is_annotation_key_text(text):
    normalized = " ".join(text.strip().split())

    if not normalized:
        return False

    letters = re.findall(r"[A-Za-z]", normalized)

    if not letters:
        return False

    if normalized.startswith("("):
        digit_count = sum(char.isdigit() for char in normalized)
        return digit_count < len(letters) * 2

    return normalized.count("(") >= 2 or normalized.count(")") >= 2


def _is_probable_footer_text(text):
    lowered = " ".join(text.lower().split())

    if lowered.startswith("revised april"):
        return True

    if lowered.startswith("please include all relevant clinical documentation"):
        return True

    if "page" in lowered and " of " in lowered:
        return True

    if lowered.startswith("form ") and "patient:" in lowered:
        return True

    return False


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


def _collect_line_key_groups(line, max_gap=45):
    groups = []
    items = line["items"]
    idx = 0

    while idx < len(items):
        item = items[idx]

        if not _looks_like_key(item):
            idx += 1
            continue

        group_items = [item]
        cursor = idx + 1

        while cursor < len(items):
            next_item = items[cursor]
            gap = next_item["bbox"][0] - group_items[-1]["bbox"][2]

            if gap > max_gap:
                break

            if _looks_like_value(next_item) and not _looks_like_key_connector(next_item):
                break

            if _looks_like_key(next_item) or _looks_like_key_connector(next_item):
                group_items.append(next_item)
                cursor += 1
                continue

            break

        groups.append({
            "text": _merge_text(group_items),
            "bbox": _union_bbox(group_items),
            "items": group_items,
        })
        idx = cursor

    return groups


def _split_line_blocks(items, gap_threshold):
    blocks = []

    for item in sorted(items, key=lambda value: (value["bbox"][0], value["bbox"][1])):
        if not blocks:
            blocks.append([item])
            continue

        gap = item["bbox"][0] - blocks[-1][-1]["bbox"][2]

        if gap > gap_threshold:
            blocks.append([item])
            continue

        blocks[-1].append(item)

    return blocks


def _collect_annotation_key_groups(line, line_threshold):
    items = [
        item for item in line["items"]
        if item["text"].strip() and item["text"].strip() != "|" and not is_section(item)
    ]

    if not items:
        return []

    blocks = _split_line_blocks(items, max(45, int(line_threshold * 2.5)))
    groups = []

    for block in blocks:
        text = _merge_text(block)

        if not _is_annotation_key_text(text):
            continue

        groups.append({
            "text": text,
            "bbox": _union_bbox(block),
            "items": block,
        })

    return groups if len(groups) >= 2 else []


def _collect_value_groups(line, line_threshold):
    items = [
        item for item in line["items"]
        if item["text"].strip()
        and item["text"].strip() != "|"
        and not is_section(item)
        and not _looks_like_key(item)
        and not _looks_like_key_connector(item)
    ]

    if not items:
        return []

    blocks = _split_line_blocks(items, max(40, int(line_threshold * 2.35)))

    return [
        {
            "text": clean_value(_merge_text(block)),
            "bbox": _union_bbox(block),
            "items": block,
        }
        for block in blocks
        if clean_value(_merge_text(block))
    ]


def _non_key_line_items(line, key_groups):
    key_item_ids = {id(item) for group in key_groups for item in group["items"]}
    return [
        item for item in line["items"]
        if id(item) not in key_item_ids and not is_section(item)
    ]


def _score_vertical_pair(key_group, value_group):
    key_bbox = key_group["bbox"]
    value_bbox = value_group["bbox"]
    overlap = max(0, min(key_bbox[2], value_bbox[2]) - max(key_bbox[0], value_bbox[0]))
    center_delta = abs(_cx(key_bbox) - _cx(value_bbox))

    if overlap == 0 and center_delta > max(280, _width(key_bbox) * 0.85):
        return None

    return center_delta - (overlap * 0.35)


def _extract_above_key_value_pairs(lines, line_idx, current_section, line_threshold):
    if line_idx == 0:
        return []

    line = lines[line_idx]
    key_groups = _collect_annotation_key_groups(line, line_threshold)

    if not key_groups:
        return []

    previous_line = lines[line_idx - 1]

    if _extract_section_from_line(previous_line):
        return []

    if _collect_line_key_groups(previous_line):
        return []

    value_groups = _collect_value_groups(previous_line, line_threshold)

    if len(value_groups) < 2:
        return []

    records = []
    used_value_indices = set()
    section_name = current_section or "GENERAL"

    for key_group in key_groups:
        best_match = None

        for value_idx, value_group in enumerate(value_groups):
            if value_idx in used_value_indices:
                continue

            score = _score_vertical_pair(key_group, value_group)

            if score is None:
                continue

            if best_match is None or score < best_match[0]:
                best_match = (score, value_idx, value_group)

        if best_match is None:
            continue

        _, value_idx, value_group = best_match
        used_value_indices.add(value_idx)
        key = clean_key(key_group["text"])

        if not key:
            continue

        records.append({
            "section": section_name,
            "key": key,
            "value": value_group["text"],
        })

    return records if len(records) >= 2 else []


def _collect_contiguous_items(items, max_gap):
    if not items:
        return []

    contiguous = [items[0]]

    for item in items[1:]:
        gap = item["bbox"][0] - contiguous[-1]["bbox"][2]

        if gap > max_gap:
            break

        contiguous.append(item)

    return contiguous


def _filter_value_items(items, left_boundary, right_boundary):
    candidates = []

    for item in items:
        text = item["text"].strip()

        if _is_noise_value(text):
            continue

        if item["bbox"][2] <= left_boundary or item["bbox"][0] >= right_boundary:
            continue

        if not (_looks_like_value(item) or item.get("label", "O") == "O"):
            continue

        candidates.append(item)

    return candidates


def _join_value_items(items):
    return clean_value(_merge_text(items))


def _collect_inline_value(line, key_groups, key_index, line_threshold):
    key_group = key_groups[key_index]
    right_boundary = key_groups[key_index + 1]["bbox"][0] - 20 if key_index + 1 < len(key_groups) else float("inf")
    max_gap = max(120, line_threshold * 8)
    candidates = _filter_value_items(
        _non_key_line_items(line, key_groups),
        key_group["bbox"][2] - 5,
        right_boundary,
    )

    if not candidates:
        return "", None

    contiguous = _collect_contiguous_items(candidates, max_gap)
    return _join_value_items(contiguous), contiguous


def _collect_below_value(lines, start_line_idx, key_groups, key_index, line_threshold, anchor_x0=None):
    key_group = key_groups[key_index]
    right_boundary = key_groups[key_index + 1]["bbox"][0] - 20 if key_index + 1 < len(key_groups) else float("inf")
    left_boundary = (anchor_x0 - 40) if anchor_x0 is not None else (key_group["bbox"][0] - 40)
    value_lines = []
    max_vertical_gap = max(34, int(line_threshold * 2.5))
    previous_bottom = lines[start_line_idx]["bbox"][3]

    for line_idx in range(start_line_idx + 1, len(lines)):
        line = lines[line_idx]
        vertical_gap = line["bbox"][1] - previous_bottom

        if vertical_gap > max_vertical_gap:
            break

        if _extract_section_from_line(line):
            break

        if _collect_line_key_groups(line):
            break

        candidates = _filter_value_items(line["items"], left_boundary, right_boundary)

        if not candidates:
            if value_lines:
                break

            previous_bottom = line["bbox"][3]
            continue

        if value_lines:
            anchor_x0 = min(item["bbox"][0] for item in value_lines[0])
            candidates = [
                item for item in candidates
                if item["bbox"][2] >= anchor_x0 - 25
            ]

        contiguous = _collect_contiguous_items(candidates, max(140, line_threshold * 9))

        if not contiguous:
            break

        value_lines.append(contiguous)
        previous_bottom = line["bbox"][3]

    if not value_lines:
        return ""

    return "\n".join(_join_value_items(items) for items in value_lines if items).strip()


# =========================
# KV PAIRING
# =========================

def extract_key_value_pairs(merged):
    lines = _build_lines(merged)
    records = []
    current_section = None
    line_threshold = _estimate_line_threshold(merged)
    previous_was_section = False

    for line_idx, line in enumerate(lines):
        line_text = clean_value(_merge_text(line["items"]))

        if _is_probable_footer_text(line_text):
            continue

        section_name = _extract_section_from_line(line)

        if section_name:
            current_section = section_name
            previous_was_section = True
            continue

        vertical_records = _extract_above_key_value_pairs(lines, line_idx, current_section, line_threshold)

        if vertical_records:
            records.extend(vertical_records)
            previous_was_section = False
            continue

        key_groups = _collect_line_key_groups(line)

        if not key_groups:
            if (
                current_section
                and previous_was_section
                and line_text
                and not any(item.get("label") in OPTION_LABELS for item in line["items"])
                and len(line_text.split()) >= 3
            ):
                records.append({
                    "section": current_section,
                    "key": "content",
                    "value": line_text,
                })

            previous_was_section = False
            continue

        section_name = current_section or "GENERAL"
        previous_was_section = False

        for key_index, key_group in enumerate(key_groups):
            key = clean_key(key_group["text"])

            if not key:
                continue

            value, inline_items = _collect_inline_value(line, key_groups, key_index, line_threshold)

            if not value:
                value = _collect_below_value(lines, line_idx, key_groups, key_index, line_threshold)
            else:
                inline_anchor_x0 = min(item["bbox"][0] for item in inline_items) if inline_items else None
                continuation = _collect_below_value(
                    lines,
                    line_idx,
                    key_groups,
                    key_index,
                    line_threshold,
                    anchor_x0=inline_anchor_x0,
                )

                if continuation:
                    value = f"{value}\n{continuation}"

            if not value and not key.endswith(":") and len(key_groups) > 1:
                continue

            records.append({
                "section": section_name,
                "key": key,
                "value": value,
            })

    return records


# =========================
# STRUCTURED OUTPUT
# =========================

def extract_structured(words_data, labels=None):
    if labels:
        for word, label in zip(words_data, labels):
            word["label"] = label

    merged = merge_words(words_data)
    structured = OrderedDict()

    for record in extract_key_value_pairs(merged):
        section = record["section"]
        structured.setdefault(section, OrderedDict())
        key = record["key"]

        if key == "content" and key in structured[section]:
            existing = structured[section][key].strip()
            addition = record["value"].strip()

            if addition and addition not in existing.splitlines():
                structured[section][key] = f"{existing}\n{addition}".strip()

            continue

        if key not in structured[section]:
            structured[section][key] = record["value"]
            continue

        suffix = 2

        while f"{key} ({suffix})" in structured[section]:
            suffix += 1

        structured[section][f"{key} ({suffix})"] = record["value"]

    return OrderedDict((section, values) for section, values in structured.items() if values)
