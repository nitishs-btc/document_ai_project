def merge_words_linewise(words_data, labels):
    merged = []

    for w, label in zip(words_data, labels):
        text = w["text"]
        bbox = w["bbox"]

        if not merged:
            merged.append({"text": text, "bbox": bbox.copy(), "label": label})
            continue

        prev = merged[-1]

        px0, py0, px1, py1 = prev["bbox"]
        cx0, cy0, cx1, cy1 = bbox

        same_line = abs((py0 + py1)/2 - (cy0 + cy1)/2) < 15

        if same_line and label == prev["label"]:
            prev["text"] += " " + text
            prev["bbox"][2] = cx1
        else:
            merged.append({"text": text, "bbox": bbox.copy(), "label": label})

    return merged

def extract_key_value(words_data, labels):

    merged = merge_words_linewise(words_data, labels)

    keys = [m for m in merged if m["label"] == "B-KEY"]
    values = [m for m in merged if m["label"] == "B-VALUE"]

    result = {}

    for key in keys:
        kx0, ky0, kx1, ky1 = key["bbox"]
        kcx = (kx0 + kx1) / 2
        kcy = (ky0 + ky1) / 2

        best_value = None
        best_score = float("inf")

        for val in values:
            vx0, vy0, vx1, vy1 = val["bbox"]
            vcx = (vx0 + vx1) / 2
            vcy = (vy0 + vy1) / 2

            dx = vcx - kcx
            dy = vcy - kcy

            abs_dx = abs(dx)
            abs_dy = abs(dy)

            score = float("inf")

            # ==========================
            # 1️⃣ RIGHT SIDE (primary)
            # ==========================
            if dx > 0 and abs_dy < 20:
                score = abs_dx

            # ==========================
            # 2️⃣ BELOW (stacked layout)
            # ==========================
            elif dy > 0 and abs_dx < 150 and abs_dy < 80:
                score = abs_dx + abs_dy * 2

            # ==========================
            # 3️⃣ ABOVE (🔥 FIX FOR YOUR ISSUE)
            # ==========================
            elif dy < 0 and abs_dx < 150 and abs_dy < 80:
                score = abs_dx + abs_dy * 2.5

            # ==========================
            # 4️⃣ SAME COLUMN (table fix)
            # ==========================
            elif abs_dx < 40 and abs_dy < 200:
                score = abs_dy + abs_dx * 3

            if score < best_score:
                best_score = score
                best_value = val["text"]

        result[key["text"]] = best_value if best_value else ""

    return result