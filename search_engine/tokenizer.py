from underthesea import word_tokenize
import re

def clean_text(text):
    # Loại bỏ các ký tự đặc biệt, giữ lại chữ, số và khoảng trắng
    text = re.sub(r"[^\w\s]", " ", text)
    # Bỏ các ký tự xuống dòng, tab và chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text):
    cleaned = clean_text(text)
    return word_tokenize(cleaned, format="text").lower().split()
