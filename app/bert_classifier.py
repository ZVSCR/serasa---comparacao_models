import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "app/model"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()


def classify_text(text: str):
    """
    Retorna:
      pred: 0 (fake) ou 1 (real)
      confidence: probabilidade máxima, já no formato que sua API usa
    """
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

    return pred, confidence
