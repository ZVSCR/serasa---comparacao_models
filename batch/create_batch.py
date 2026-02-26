import json
from config import CHATGPT_MODEL
from schemas import LLMClassification
from debate.prompts import SYSTEM_PROMPT

def build_batch_file(dataset):

    with open("batch_input.jsonl", "w", encoding="utf-8") as f:
        for i, text in enumerate(dataset):

            entry = {
                "custom_id": f"news-{i}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": CHATGPT_MODEL,
                    "input": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": LLMClassification.model_json_schema()
                    }
                }
            }

            f.write(json.dumps(entry) + "\n")