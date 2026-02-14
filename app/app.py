from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.bert_classifier import classify_text
from app.ocr import extract_text_from_image
from app.preprocess import preprocess_text


app = FastAPI(
    title="API Fake News — BERT Fine-Tuned",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos de entrada/saída
class TextInput(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "API Fake News — BERT Fine-Tuned ativa!"}

# OCR
@app.post("/img-to-txt")
async def img_to_txt(file: UploadFile = File(...)):
    bytes_data = await file.read()
    text = extract_text_from_image(bytes_data)
    return {"text": text}

# Predição
@app.post("/predict")
async def predict_text(input: TextInput):
    text = preprocess_text(input.text)

    if len(text) < 20:
        return {
            "prediction": "Indefinido",
            "confidence": 0.0,
            "message": "O texto é muito curto para análise confiável."
        }

    pred, conf = classify_text(text)

    if pred == 1:
        label = "Notícia Real"

        if conf > 0.90:
            message = "Essa notícia parece altamente confiável."
        elif conf > 0.75:
            message = "Provável notícia real, mas é bom conferir as fontes."
        else:
            message = "O modelo pende para real, mas com baixa confiança."
    else:
        label = "Fake News"

        if conf > 0.90:
            message = "Forte indicação de que esta notícia é falsa."
        elif conf > 0.75:
            message = "Provável conteúdo falso, mas recomenda-se verificar fontes."
        else:
            message = "O modelo pende para falsa, mas sem alta confiança."

    return {
        "prediction": label,
        "confidence": round(conf, 3),
        "message": message
    }
