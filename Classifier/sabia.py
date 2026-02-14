# classifiers/sabia.py

import requests
import json
from datetime import datetime
from config import SABIA_API_KEY, SABIA_ENDPOINT, TEMPERATURE
from debate.prompts import SYSTEM_CLASSIFY

def classify_sabia(text: str):

    payload = {
        "prompt": SYSTEM_CLASSIFY + "\n\n" + text,
        "temperature": TEMPERATURE
    }

    headers = {
        "Authorization": f"Bearer {SABIA_API_KEY}",
        "Content-Type": "application/json"
    }

    r = requests.post(SABIA_ENDPOINT, json=payload, headers=headers)

    raw = r.json()["response"]
    parsed = json.loads(raw)

    return {
        "model": "sabia",
        **parsed,
        "raw_response": raw,
        "timestamp": datetime.utcnow().isoformat()
    }
