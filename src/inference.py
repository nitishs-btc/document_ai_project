import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForTokenClassification

DEBUG_MODEL_DIR = "debug/model"
os.makedirs(DEBUG_MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.abspath("model/final_model")

processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, local_files_only=True)

model.eval()


def normalize_bbox(bbox, width, height):
    x0, y0, x1, y1 = bbox

    return [
        max(0, min(1000, int(1000 * x0 / width))),
        max(0, min(1000, int(1000 * y0 / height))),
        max(0, min(1000, int(1000 * x1 / width))),
        max(0, min(1000, int(1000 * y1 / height)))
    ]


def predict(image_path, words_data):

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    tokens = []
    boxes = []

    for w in words_data:
        if not w.get("text") or not w.get("bbox"):
            continue

        tokens.append(w["text"])
        boxes.append(normalize_bbox(w["bbox"], width, height))

    encoding = processor(
        images=image,
        text=tokens,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length"
    )

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()

    word_ids = encoding.word_ids()
    id2label = model.config.id2label

    labels = []
    seen = set()

    debug_output = []

    for pred, word_id in zip(predictions, word_ids):
        if word_id is None or word_id in seen:
            continue

        label = id2label[pred]
        labels.append(label)

        debug_output.append({
            "text": tokens[word_id],
            "bbox": boxes[word_id],
            "label": label
        })

        seen.add(word_id)

    # 🔥 SAVE MODEL OUTPUT
    filename = os.path.basename(image_path).replace(".png", ".json")
    with open(os.path.join(DEBUG_MODEL_DIR, filename), "w") as f:
        json.dump(debug_output, f, indent=2)

    return tokens, labels