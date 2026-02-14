import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import numpy as np
import random

from utils.preprocess import preprocess_text

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
OUTPUT_DIR = "app/model"

FAKE_DIR = "data/fake_news/financeiros"
REAL_DIR = "data/real_news/financeiros"

EPOCHS = 3
BATCH_SIZE = 8
LR = 2e-5
MAX_LEN = 256

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Treinando em: {device}")

def load_texts_from_dir(directory, label):
    samples = []

    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".txt"):
                path = os.path.join(root, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = preprocess_text(f.read())
                        samples.append((text, label))
                except Exception as e:
                    print(f"Erro ao ler {path}: {e}")

    return samples


def load_dataset():
    fake = load_texts_from_dir(FAKE_DIR, 0)
    real = load_texts_from_dir(REAL_DIR, 1)

    all_data = fake + real
    random.shuffle(all_data)

    print(f"Fake: {len(fake)} | Real: {len(real)} | Total: {len(all_data)}")

    texts, labels = zip(*all_data)
    return list(texts), list(labels)

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        encoded = {k: v.squeeze() for k, v in encoded.items()}
        encoded["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return encoded

def train():
    texts, labels = load_dataset()

    # SEPARAÇÃO REAL entre treino, validação e teste
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, stratify=labels, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.10, stratify=y_train, random_state=42
    )

    print(f"\nTreino: {len(X_train)} | Val: {len(X_val)} | Teste: {len(X_test)}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    # LOADERS
    train_dataset = NewsDataset(X_train, y_train, tokenizer)
    val_dataset = NewsDataset(X_val, y_val, tokenizer)
    test_dataset = NewsDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    print("Iniciando fine-tuning...\n")

    model.train()

    for epoch in range(EPOCHS):
        print(f"=== Época {epoch+1}/{EPOCHS} ===")
        epoch_loss = 0

        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        print(f"Loss da época: {epoch_loss / len(train_loader):.4f}")

    print("\nAvaliando...")

    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            labels = batch["labels"].numpy()
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_true.extend(labels)

    # MÉTRICAS
    print("\n=== Classification Report ===")
    print(classification_report(all_true, all_preds, target_names=["Fake", "Real"]))
    print("Accuracy:", accuracy_score(all_true, all_preds))

    # SALVAR MODELO
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nModelo salvo em: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    train()
