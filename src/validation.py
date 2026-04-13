import re


def clean_key(key):
    parts = key.split()
    unique = []

    for p in parts:
        if p not in unique:
            unique.append(p)

    return " ".join(unique)


def extract_email(text):
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else text


def clean_phone(text):
    return re.sub(r'(\d)\s+\1', r'\1', text)


def is_valid_key(key):
    if len(key) < 3:
        return False

    if key.lower() in ["aa", "yt", "hats"]:
        return False

    return True


def validate_json(data):

    if not isinstance(data, dict):
        return data

    cleaned = {}

    for k, v in data.items():

        key = clean_key(k)

        if not is_valid_key(key):
            continue

        if isinstance(v, str):
            v = v.strip()

            if "@" in v:
                v = extract_email(v)

            if any(char.isdigit() for char in v):
                v = clean_phone(v)

        cleaned[key] = v

    return cleaned


def final_cleanup(data):
    return validate_json(data)