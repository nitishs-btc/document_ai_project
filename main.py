import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
from collections import OrderedDict
from pathlib import Path

from src.ocr import run_ocr
from src.inference import predict
from src.utils import extract_structured, sort_words
from src.pdf_service import PDFService
from src.checkbox import map_checkbox_to_text, classify_checkbox, detect_checkboxes, group_checkbox_results
from src.io_utils import group_image_paths, image_output_stem

OUTPUT_FOLDER = "output"
TEMP_IMAGE_FOLDER = "input"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_IMAGE_FOLDER, exist_ok=True)

IMAGE_EXT = (".png", ".jpg", ".jpeg")
PDF_EXT = (".pdf",)


# ==========================
# PROCESS IMAGE
# ==========================
def process_image(img_path):
    print(f"\n🖼️ Processing Image: {img_path}")

    # OCR
    words_data = run_ocr(img_path)
    print(f"🔍 OCR words: {len(words_data)}")

    if not words_data:
        print("⚠️ No OCR output")
        return None

    words_data = sort_words(words_data)

    words = [w["text"] for w in words_data]
    boxes = [w["bbox"] for w in words_data]

    # MODEL PREDICTION
    predictions = predict(img_path, words, boxes)

    # =========================
    # 🔥 FIX: SAFE LABEL MAPPING (bbox based)
    # =========================
    for w in words_data:
        w["label"] = "O"

    for p in predictions:
        for w in words_data:
            if w["bbox"] == p["bbox"]:
                w["label"] = p["label"]
                break

    # DEBUG
    os.makedirs("debug", exist_ok=True)
    with open("debug/predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    with open("debug/words_with_labels.json", "w") as f:
        json.dump(words_data, f, indent=2)

    # STRUCTURED OUTPUT
    labels = [w["label"] for w in words_data]
    structured_data = extract_structured(words_data, labels)

    # CHECKBOX
    checkbox_boxes = detect_checkboxes(img_path, words_data)
    checkbox_states = classify_checkbox(img_path, checkbox_boxes)
    checkbox_results = map_checkbox_to_text(checkbox_states, words_data, img_path=img_path)

    return {
        "form_data": structured_data,
        "checkboxes": checkbox_results,
        "checkbox_groups": group_checkbox_results(checkbox_results)
    }


def process_image_group(image_paths):
    if len(image_paths) == 1:
        return process_image(image_paths[0])

    pages = OrderedDict()

    for image_path in image_paths:
        result = process_image(image_path)

        if result:
            pages[Path(image_path).name] = result

    return {"pages": pages}


# ==========================
# PDF PROCESS
# ==========================
def process_pdf(pdf_path, pdf_service):
    print(f"\n📄 Processing PDF: {pdf_path}")

    images_map = pdf_service.convert_pdfs([pdf_path])
    pdf_name = list(images_map.keys())[0]

    pdf_result = {}

    for img_path in images_map[pdf_name]:
        result = process_image(img_path)
        if result:
            page_name = os.path.basename(img_path)
            pdf_result[page_name] = result

    return pdf_name, pdf_result


# ==========================
# MAIN
# ==========================
if len(sys.argv) < 2:
    print("❌ Usage: python main.py <image | pdf | folder>")
    exit()

input_path = sys.argv[1]
pdf_service = PDFService(TEMP_IMAGE_FOLDER)

if os.path.isfile(input_path) and input_path.lower().endswith(IMAGE_EXT):
    result = process_image(input_path)

    if result:
        name = image_output_stem(input_path)
        output_path = os.path.join(OUTPUT_FOLDER, f"{name}.json")

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved: {output_path}")

elif os.path.isfile(input_path) and input_path.lower().endswith(PDF_EXT):
    pdf_name, pdf_result = process_pdf(input_path, pdf_service)

    output_path = os.path.join(OUTPUT_FOLDER, f"{pdf_name}.json")

    with open(output_path, "w") as f:
        json.dump(pdf_result, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved: {output_path}")

elif os.path.isdir(input_path):
    print(f"\n📂 Processing Folder: {input_path}")

    image_paths = [
        os.path.join(input_path, file_name)
        for file_name in os.listdir(input_path)
        if file_name.lower().endswith(IMAGE_EXT)
    ]

    for document in group_image_paths(image_paths):
        result = process_image_group(document["paths"])

        if result:
            output_path = os.path.join(OUTPUT_FOLDER, f"{document['name']}.json")

            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"✅ Saved: {output_path}")
