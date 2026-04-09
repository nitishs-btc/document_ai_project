import cv2
import pytesseract


def run_ocr(image_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # Better OCR config
    custom_config = r'--oem 3 --psm 6'

    data = pytesseract.image_to_data(
        img,
        config=custom_config,
        output_type=pytesseract.Output.DICT
    )

    words_data = []

    for i in range(len(data['text'])):
        text = data['text'][i].strip()

        if text == "":
            continue

        x = data['left'][i]
        y = data['top'][i]
        width = data['width'][i]
        height = data['height'][i]

        x0 = x
        y0 = y
        x1 = x + width
        y1 = y + height

        words_data.append({
            "text": text,
            "bbox": [x0, y0, x1, y1]
        })

    return words_data