# utils/logger.py

import json

def log_result(result):

    with open("logs/results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
