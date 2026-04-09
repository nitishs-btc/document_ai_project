import os
import json
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

ocr_model = ocr_predictor(pretrained=True)

DEBUG_OCR_DIR = "debug/ocr"
os.makedirs(DEBUG_OCR_DIR, exist_ok=True)


def run_ocr(image_path):

    doc = DocumentFile.from_images(image_path)
    result = ocr_model(doc)

    words = []

    for page in result.pages:
        h, w = page.dimensions

        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    (x0, y0), (x1, y1) = word.geometry

                    words.append({
                        "text": word.value,
                        "bbox": [
                            int(x0 * w),
                            int(y0 * h),
                            int(x1 * w),
                            int(y1 * h)
                        ]
                    })

    # 🔥 SAVE DEBUG OCR
    filename = os.path.basename(image_path).replace(".png", ".json")
    with open(os.path.join(DEBUG_OCR_DIR, filename), "w") as f:
        json.dump(words, f, indent=2)

    return words