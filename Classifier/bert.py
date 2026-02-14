# classifiers/bert.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime

MODEL_DIR = "app/model"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

LABEL_MAP = {0: "fake", 1: "real"}

def classify_bert(text: str):

    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = model(**encoded)
        logits = out.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred = int(probs.argmax())
    confidence = float(probs.max())

    return {
        "model": "bert_local",
        "prediction": pred,
        "label_name": LABEL_MAP[pred],
        "confidence": confidence,
        "timestamp": datetime.utcnow().isoformat()
    }
