from openai import OpenAI
from datetime import datetime
from config import CHATGPT_API_KEY, CHATGPT_MODEL
from schemas import LLMClassification
from debate.prompts import SYSTEM_PROMPT

client = OpenAI(api_key=CHATGPT_API_KEY)

def classify_chatgpt(text: str):

    response = client.responses.parse(
        model=CHATGPT_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        response_format=LLMClassification
    )

    parsed: LLMClassification = response.output_parsed

    return {
        "model": "chatgpt",
        "justification": parsed.justification,
        "prediction": parsed.prediction,
        "label_name": parsed.label_name,
        "timestamp": datetime.utcnow().isoformat()
    }