import pytesseract
from PIL import Image
from io import BytesIO

def extract_text_from_image(image_bytes):
    img = Image.open(BytesIO(image_bytes))
    return pytesseract.image_to_string(img, lang="por")
