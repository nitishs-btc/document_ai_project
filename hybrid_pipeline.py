import json
import os

from src.ocr import run_ocr
from src.inference import predict

from src.light_processing import prepare_tokens
from src.llm_mistral import run_mistral
from src.validation import final_cleanup


OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_image(image_path):
    print(f"\n🖼️ Processing: {image_path}")

    # -------------------------
    # STEP 1: OCR
    # -------------------------
    print("👉 Running OCR...")
    words_data = run_ocr(image_path)
    print("✅ OCR done:", len(words_data))

    if not words_data:
        print("❌ OCR failed")
        return None

    words = [w["text"] for w in words_data]
    boxes = [w["bbox"] for w in words_data]

    # -------------------------
    # STEP 2: LayoutLM
    # -------------------------
    print("👉 Running LayoutLM...")
    predictions = predict(image_path, words, boxes)
    print("✅ LayoutLM done:", len(predictions))

    # Map labels back
    for w in words_data:
        w["label"] = "O"

    for p in predictions:
        for w in words_data:
            if w["bbox"] == p["bbox"]:
                w["label"] = p["label"]
                break

    # -------------------------
    # STEP 3: Light Processing
    # -------------------------
    tokens = prepare_tokens(words_data)[:30]

    # -------------------------
    # STEP 4: Mistral
    # -------------------------
    print("👉 Calling Mistral...")
    structured_json = run_mistral(tokens)
    print("✅ Mistral done")

    # -------------------------
    # STEP 5: Validation
    # -------------------------
    final_json = final_cleanup(structured_json)

    return final_json


def process_folder(input_folder):
    results = {}

    for file in os.listdir(input_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_folder, file)

            result = process_image(path)
            results[file] = result

    # Save output
    output_path = os.path.join(OUTPUT_DIR, "final_output.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Output saved: {output_path}")


if __name__ == "__main__":
    input_folder = "input"
    process_folder(input_folder)