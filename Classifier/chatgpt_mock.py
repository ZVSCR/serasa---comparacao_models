from datetime import datetime

def classify_chatgpt(text: str):

    if "golpe" in text.lower():
        prediction = 0
        label = "fake"
        justification = "O texto contém termos típicos de desinformação."
    else:
        prediction = 1
        label = "real"
        justification = "O texto apresenta estrutura jornalística plausível."

    return {
        "model": "chatgpt",
        "justification": justification,
        "prediction": prediction,
        "label_name": label,
        "timestamp": datetime.utcnow().isoformat()
    }