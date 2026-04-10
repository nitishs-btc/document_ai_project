import cv2
import pytesseract
import json
import os

DEBUG_OCR_DIR = "debug/ocr"
os.makedirs(DEBUG_OCR_DIR, exist_ok=True)


def run_ocr(image_path):
    img = cv2.imread(image_path)

    # 🔥 BETTER CONFIG
    custom_config = r'--oem 3 --psm 4'

    data = pytesseract.image_to_data(
        img,
        config=custom_config,
        output_type=pytesseract.Output.DICT
    )

    words_data = []
    debug_output = []

    for i in range(len(data['text'])):
        text = data['text'][i].strip()

        if text == "":
            continue

        conf = int(data['conf'][i])

        if conf < 40:  # 🔥 stricter filtering
            continue

        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]

        bbox = [x, y, x + w, y + h]

        word = {
            "text": text,
            "bbox": bbox
        }

        words_data.append(word)

        debug_output.append({
            "text": text,
            "bbox": bbox,
            "conf": conf
        })

    filename = os.path.basename(image_path).replace(".png", ".json")

    with open(os.path.join(DEBUG_OCR_DIR, filename), "w") as f:
        json.dump(debug_output, f, indent=2)

    return words_data