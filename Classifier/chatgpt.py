# classifiers/chatgpt.py

import json
from datetime import datetime
from openai import OpenAI
from config import CHATGPT_API_KEY, CHATGPT_MODEL, TEMPERATURE
from debate.prompts import SYSTEM_CLASSIFY

client = OpenAI(api_key=CHATGPT_API_KEY)

def classify_chatgpt(text: str):

    response = client.chat.completions.create(
        model=CHATGPT_MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_CLASSIFY},
            {"role": "user", "content": text}
        ]
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    return {
        "model": "chatgpt",
        **parsed,
        "raw_response": raw,
        "timestamp": datetime.utcnow().isoformat()
    }
