import requests
import json
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"


def clean_llm_output(text):
    if not text:
        return text

    # remove ```json blocks
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    # trim
    text = text.strip()

    # try to extract JSON part only
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return text


def run_mistral(tokens, draft_json=None):

    if draft_json is None:
        draft_json = {}

    prompt = f"""
Convert the following tokens into valid JSON.

Rules:
- Output MUST be valid JSON
- Do NOT use ```json
- Do NOT explain
- Do NOT truncate

DATA:
{json.dumps(tokens)}

Return only JSON.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 120
            }
        }
    )

    result = response.json()["response"]
    cleaned = clean_llm_output(result)

    try:
        return json.loads(cleaned)
    except Exception:
        print("\n⚠️ JSON parse failed. Raw output:\n")
        print(cleaned[:500])
        return {"error": "Invalid JSON", "raw": cleaned}