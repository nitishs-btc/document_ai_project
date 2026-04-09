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

        same_line = abs((py0 + py1) / 2 - (cy0 + cy1) / 2) < 15

        # 🔥 FIX: merge consecutive KEY tokens also
        if same_line and (
            label == prev["label"] or
            (label == "B-KEY" and prev["label"] == "B-KEY")
        ):
            prev["text"] += " " + text
            prev["bbox"][2] = cx1
        else:
            merged.append({"text": text, "bbox": bbox.copy(), "label": label})

    return merged


def _row_height(bbox):
    return bbox[3] - bbox[1]


def extract_key_value(words_data, labels):

    merged = merge_words_linewise(words_data, labels)

    result = {}
    current_section = None

    for i, item in enumerate(merged):

        # =========================
        # SECTION
        # =========================
        if item["label"] == "B-SECTION":
            current_section = item["text"].strip()
            if current_section not in result:
                result[current_section] = {}
            continue

        if item["label"] != "B-KEY":
            continue

        key = item
        kx0, ky0, kx1, ky1 = key["bbox"]
        kcx = (kx0 + kx1) / 2
        kcy = (ky0 + ky1) / 2
        k_height = max(_row_height(key["bbox"]), 12)

        best_value = None

        # =====================================================
        # 🚀 1. SAFE ROW MAPPING (STRICT VERSION)
        # =====================================================
        above_values = []
        same_row_keys = []

        for itm in merged:
            vx0, vy0, vx1, vy1 = itm["bbox"]
            vcx = (vx0 + vx1) / 2
            vcy = (vy0 + vy1) / 2

            if itm["label"] == "B-VALUE":
                if 0 < (ky0 - vy1) < 100:
                    above_values.append(itm)

            if itm["label"] == "B-KEY":
                if abs(vcy - kcy) < k_height:
                    same_row_keys.append(itm)

        # 🔥 apply ONLY if counts match (important fix)
        if len(above_values) == len(same_row_keys) and len(above_values) >= 3:

            above_values.sort(key=lambda x: x["bbox"][0])
            same_row_keys.sort(key=lambda x: x["bbox"][0])

            for idx, k in enumerate(same_row_keys):
                val = above_values[idx]["text"]

                if current_section:
                    result[current_section][k["text"]] = val
                else:
                    result[k["text"]] = val

            continue

        # =====================================================
        # 2. SAME ROW (RELAXED FIX)
        # =====================================================
        same_row = []

        for val in merged:
            if val["label"] != "B-VALUE":
                continue

            vx0, vy0, vx1, vy1 = val["bbox"]
            vcy = (vy0 + vy1) / 2

            if abs(vcy - kcy) < k_height:
                dx = vx0 - kx1

                if dx >= 0 and dx < 400:   # 🔥 relaxed
                    same_row.append((dx, val))

        if same_row:
            same_row.sort(key=lambda x: x[0])
            best_value = same_row[0][1]["text"]

        # =====================================================
        # 3. ABOVE (COLUMN OVERLAP FIX)
        # =====================================================
        if not best_value:

            best_overlap = 0
            best_val = None

            for val in merged:
                if val["label"] != "B-VALUE":
                    continue

                vx0, vy0, vx1, vy1 = val["bbox"]

                # must be above
                if not (0 < (ky0 - vy1) < 150):
                    continue

                # overlap instead of center distance
                overlap = min(kx1, vx1) - max(kx0, vx0)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_val = val["text"]

            if best_val:
                best_value = best_val

        # =====================================================
        # 4. BELOW (MULTI-LINE SAFE)
        # =====================================================
        if not best_value:
            collected = []

            for j in range(i + 1, len(merged)):
                nxt = merged[j]

                if nxt["label"] == "B-KEY":
                    break

                if nxt["label"] not in ["B-VALUE", "O"]:
                    continue

                vx0, vy0, vx1, vy1 = nxt["bbox"]

                if vy0 <= ky1:
                    continue

                text = nxt["text"].strip()
                if len(text) < 2:
                    continue

                collected.append((vy0, text))

            if collected:
                collected.sort(key=lambda x: x[0])
                best_value = " ".join(v[1] for v in collected)

        final_value = best_value.strip() if best_value else ""

        # =========================
        # STORE
        # =========================
        if current_section:
            result[current_section][key["text"]] = final_value
        else:
            result[key["text"]] = final_value

    return result