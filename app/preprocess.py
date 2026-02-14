import re
import unicodedata

def preprocess_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\wÀ-ÖØ-öø-ÿ?!,. ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

