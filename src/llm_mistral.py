import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"


def run_mistral(tokens, draft_json=None):

    if draft_json is None:
        draft_json = {}

    prompt = f"""
You are a document parser.

INPUT:
1. OCR tokens with labels:
{json.dumps(tokens)}

2. Draft JSON (may be empty or incorrect):
{json.dumps(draft_json)}

TASK:
- Create clean structured JSON
- Fix key-value pairing
- Merge multi-line values
- Remove duplicate keys
- Ignore noise (like random text)
- Handle checkboxes if present

STRICT RULES:
- Do NOT hallucinate
- Use only given data
- Output MUST be valid JSON
- No explanation

OUTPUT:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 60
            }
        }
    )

    try:
        result = response.json()["response"]
        return json.loads(result)
    except Exception as e:
        print("⚠️ LLM JSON parse failed:", e)
        return {"error": "Invalid JSON", "raw": result}