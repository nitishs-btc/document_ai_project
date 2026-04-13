import requests

url = "http://localhost:11434/api/generate"

prompt = """
Fix this JSON:
{"Phone: Phone:": "12345"}
Return only JSON.
"""

response = requests.post(url, json={
    "model": "mistral",
    "prompt": prompt,
    "stream": False
})

print(response.json()["response"])