import os
import sys
import json

from src.ocr import run_ocr
from src.inference import predict
from src.utils import extract_key_value
from src.pdf_service import PDFService

OUTPUT_FOLDER = "output"
TEMP_IMAGE_FOLDER = "input"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_IMAGE_FOLDER, exist_ok=True)

# ==========================
# Helpers
# ==========================
IMAGE_EXT = (".png", ".jpg", ".jpeg")
PDF_EXT   = (".pdf",)


def process_image(img_path):
    print(f"\n🖼️ Processing Image: {img_path}")

    words_data = run_ocr(img_path)
    print(f"🔍 OCR words: {len(words_data)}")

    if not words_data:
        print("⚠️ No OCR output")
        return None

    tokens, labels = predict(img_path, words_data)
    result = extract_key_value(words_data, labels)

    return result


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
# MAIN ENTRY
# ==========================
if len(sys.argv) < 2:
    print("❌ Usage:")
    print("   python main.py <image | pdf | folder>")
    exit()

input_path = sys.argv[1]

pdf_service = PDFService(TEMP_IMAGE_FOLDER)

# ==========================
# CASE 1: Single Image
# ==========================
if os.path.isfile(input_path) and input_path.lower().endswith(IMAGE_EXT):

    result = process_image(input_path)

    if result:
        name = os.path.basename(input_path).split(".")[0]
        output_path = os.path.join(OUTPUT_FOLDER, f"{name}.json")

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"✅ Saved: {output_path}")


# ==========================
# CASE 2: Single PDF
# ==========================
elif os.path.isfile(input_path) and input_path.lower().endswith(PDF_EXT):

    pdf_name, pdf_result = process_pdf(input_path, pdf_service)

    output_path = os.path.join(OUTPUT_FOLDER, f"{pdf_name}.json")

    with open(output_path, "w") as f:
        json.dump(pdf_result, f, indent=2)

    print(f"✅ Saved: {output_path}")


# ==========================
# CASE 3: Folder
# ==========================
elif os.path.isdir(input_path):

    print(f"\n📂 Processing Folder: {input_path}")

    for file_name in os.listdir(input_path):
        full_path = os.path.join(input_path, file_name)

        if file_name.lower().endswith(IMAGE_EXT):
            result = process_image(full_path)

            if result:
                name = os.path.basename(file_name).split(".")[0]
                output_path = os.path.join(OUTPUT_FOLDER, f"{name}.json")

                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)

                print(f"✅ Saved: {output_path}")

        elif file_name.lower().endswith(PDF_EXT):
            pdf_name, pdf_result = process_pdf(full_path, pdf_service)

            output_path = os.path.join(OUTPUT_FOLDER, f"{pdf_name}.json")

            with open(output_path, "w") as f:
                json.dump(pdf_result, f, indent=2)

            print(f"✅ Saved: {output_path}")

        else:
            print(f"⚠️ Skipping unsupported file: {file_name}")

else:
    print("❌ Invalid input path")

print("\n🚀 Done!")