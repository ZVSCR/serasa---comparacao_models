from datetime import datetime
import random

def classify_sabia(text: str):

    prediction = random.choice([0, 1])
    label = "fake" if prediction == 0 else "real"

    return {
        "model": "sabia",
        "justification": "Simulação de análise automática.",
        "prediction": prediction,
        "label_name": label,
        "timestamp": datetime.utcnow().isoformat()
    }