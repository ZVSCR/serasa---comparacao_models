import requests
import json
import re
from datetime import datetime
from config import SABIA_API_KEY, SABIA_ENDPOINT
from debate.prompts import SYSTEM_PROMPT
from schemas import LLMClassification

def classify_sabia(text: str):

    payload = {
        "prompt": SYSTEM_PROMPT + "\n\n" + text
    }

    headers = {
        "Authorization": f"Bearer {SABIA_API_KEY}",
        "Content-Type": "application/json"
    }

    r = requests.post(
        SABIA_ENDPOINT,
        json=payload,
        headers=headers,
        timeout=30
    )

    raw_response = r.json()["response"]

    clean_json = raw_response.replace("```json", "").replace("```", "").strip()
    
    # Validação via Pydantic
    parsed = LLMClassification.model_validate_json(clean_json)

    return {
        "model": "sabia",
        "justification": parsed.justification,
        "prediction": parsed.prediction,
        "label_name": parsed.label_name,
        "timestamp": datetime.utcnow().isoformat()
    }