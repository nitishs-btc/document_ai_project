from doctr.io import DocumentFile
from doctr.models import ocr_predictor

ocr_model = ocr_predictor(pretrained=True)

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

    return words