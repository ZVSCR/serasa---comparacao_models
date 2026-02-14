import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils.preprocess import preprocess_text

FAKE_DIR = "data/fake_news/financeiros"
REAL_DIR = "data/real_news/financeiros"
MODEL_DIR = "app/model"
MAX_LEN = 256
BATCH_SIZE = 8

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========= LOAD DATA =========
def load_texts(directory, label):
    samples = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".txt"):
                path = os.path.join(root, fname)
                with open(path, "r", encoding="utf-8") as f:
                    text = preprocess_text(f.read())
                    samples.append((text, label))
    return samples

def load_dataset():
    fake = load_texts(FAKE_DIR, 0)
    real = load_texts(REAL_DIR, 1)
    data = fake + real
    texts = [t for t, _ in data]
    labels = [l for _, l in data]
    return texts, labels

# ========= DATASET =========
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        enc = {k: v.squeeze() for k, v in enc.items()}
        enc["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return enc

# ========= EVALUATION =========
def evaluate():
    print("Carregando modelo...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

    print("Carregando dataset...")
    texts, labels = load_dataset()
    dataset = NewsDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    preds = []
    true_labels = []

    print("\nAvaliando...\n")

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            p = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            l = batch["labels"].cpu().numpy()

            preds.extend(p)
            true_labels.extend(l)

    # === METRICS ===
    acc = accuracy_score(true_labels, preds)
    print(f"Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=["Fake", "Real"]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, preds))


if __name__ == "__main__":
    evaluate()
