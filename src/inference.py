import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForTokenClassification

DEBUG_MODEL_DIR = "debug/model"
os.makedirs(DEBUG_MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.abspath("model/final_model")

# =========================
# LOAD MODEL ONCE
# =========================
processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, local_files_only=True)

model.eval()

id2label = model.config.id2label


# =========================
# NORMALIZE BBOX
# =========================
def normalize_bbox(bbox, width, height):
    x0, y0, x1, y1 = bbox

    return [
        max(0, min(1000, int(1000 * x0 / width))),
        max(0, min(1000, int(1000 * y0 / height))),
        max(0, min(1000, int(1000 * x1 / width))),
        max(0, min(1000, int(1000 * y1 / height)))
    ]


# =========================
# PREDICT FUNCTION (FIXED)
# =========================
def predict(image_path, words, boxes):

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # normalize boxes
    normalized_boxes = [normalize_bbox(b, width, height) for b in boxes]

    encoding = processor(
        image,
        words,
        boxes=normalized_boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length"
    )

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()

    word_ids = encoding.word_ids()

    predicted_boxes = []
    debug_data = []
    seen_word_idx = set()

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue

        if word_idx in seen_word_idx:
            continue

        seen_word_idx.add(word_idx)

        label_id = predictions[idx]
        label = id2label[label_id]

        predicted_boxes.append({
            "bbox": boxes[word_idx],
            "label": label
        })
        debug_data.append({
            "words": words[word_idx],
            "bbox": boxes[word_idx],
            "label": label
        })
    filename = os.path.basename(image_path).replace(".png", ".json")

    with open(os.path.join(DEBUG_MODEL_DIR, filename), "w") as f:
        json.dump(debug_data, f, indent=2)
    return predicted_boxes