import re

def normalize_text(text):
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    # Fix common OCR issues
    text = text.replace("Ph one", "Phone")
    text = text.replace("FirstName", "First Name")

    return text


def prepare_tokens(words_data):
    tokens = []

    for w in words_data:
        tokens.append({
            "text": normalize_text(w["text"]),
            "label": w.get("label", "O"),
            "bbox": w.get("bbox", [])
        })

    return tokens