from collections import OrderedDict


def _cy(bbox):
    return (bbox[1] + bbox[3]) / 2


def _is_same_line(a, b, threshold=12):
    return abs(_cy(a) - _cy(b)) < threshold


def _is_below(a, b):
    return b[1] > a[3]


def _h_dist(a, b):
    return b[0] - a[2]


def _v_dist(a, b):
    return b[1] - a[3]


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


# =========================
# CLEAN KEY (STRONG)
# =========================

def clean_key(text):
    t = text.strip().lower()

    bad_tokens = ["ph", "one", "im", "dp", "fa", "|"]

    if t in bad_tokens:
        return None

    if len(t) < 3:
        return None

    # remove noisy patterns
    if any(x in t for x in ["one", "lll", "|||"]):
        return None

    return text.strip()


# =========================
# STRICT SECTION DETECTION
# =========================

def is_section(text):
    text = text.strip()

    if not _is_valid_text(text):
        return False

    words = text.split()

    # only ALL CAPS allowed
    if text.isupper() and 2 <= len(words) <= 4:
        return True

    return False


# =========================
# MERGE WORDS
# =========================

def merge_words(words_data):
    merged = []

    for w in words_data:
        text = w["text"]
        bbox = w["bbox"]
        label = w.get("label", "O")

        if not merged:
            merged.append({"text": text, "bbox": bbox.copy(), "label": label})
            continue

        prev = merged[-1]

        if (
            label == prev["label"]
            and _is_same_line(prev["bbox"], bbox)
            and 0 < (bbox[0] - prev["bbox"][2]) < 40
        ):
            prev["text"] += " " + text
            prev["bbox"][2] = bbox[2]
        else:
            merged.append({"text": text, "bbox": bbox.copy(), "label": label})

    return merged


# =========================
# KV PAIRING
# =========================

def extract_key_value(merged):
    from collections import OrderedDict

    result = OrderedDict()

    i = 0
    while i < len(merged):
        item = merged[i]

        # Only process KEY labels
        if item["label"] not in ["B-KEY", "I-KEY"]:
            i += 1
            continue

        key_text = item["text"].strip()
        key_bbox = item["bbox"]

        best_value = ""

        # 🔥 LOCAL WINDOW SEARCH (VERY IMPORTANT)
        for j in range(i + 1, min(i + 7, len(merged))):
            nxt = merged[j]

            # ❌ Stop if another key appears
            if nxt["label"] in ["B-KEY", "I-KEY"]:
                break

            # Only consider VALUE labels
            if nxt["label"] in ["B-VALUE", "I-VALUE"]:
                val_bbox = nxt["bbox"]

                # ✅ Ensure it's either same line or slightly below
                cy_diff = abs((key_bbox[1] + key_bbox[3]) / 2 - (val_bbox[1] + val_bbox[3]) / 2)

                if cy_diff < 25 or val_bbox[1] > key_bbox[3]:
                    best_value = nxt["text"]
                    break

        result[key_text] = best_value
        i += 1

    return result


# =========================
# STRUCTURED OUTPUT (FINAL)
# =========================

def extract_structured(words_data, labels=None):
    from collections import OrderedDict

    # attach labels
    if labels:
        for w, l in zip(words_data, labels):
            w["label"] = l

    merged = merge_words(words_data)
    kv = extract_key_value(merged)

    structured = OrderedDict()
    current_section = None

    for item in merged:
        text = item["text"].strip()

        # 🚫 skip long paragraphs
        if len(text.split()) > 10:
            continue

        # SECTION DETECTION
        if is_section(text):
            current_section = text
            structured[current_section] = OrderedDict()
            continue

        # KEY MAPPING
        if text in kv:
            key = clean_key(text)
            if not key:
                continue

            if not current_section:
                current_section = "GENERAL"
                structured[current_section] = OrderedDict()

            structured[current_section][key] = kv[text]

    # ✅ REMOVE EMPTY SECTIONS
    structured = OrderedDict(
        (k, v) for k, v in structured.items() if v
    )

    # ✅ REMOVE DUPLICATE VALUES (IMPORTANT FIX FOR ADDRESS)
    structured = remove_duplicate_values(structured)

    return structured


def remove_duplicate_values(structured):
    seen_values = set()

    for section in structured:
        for key in list(structured[section].keys()):
            value = structured[section][key]

            if not value:
                continue

            # If same value already seen → remove duplicate
            if value in seen_values:
                structured[section][key] = ""
            else:
                seen_values.add(value)

    return structured