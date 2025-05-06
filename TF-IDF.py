import os
import math
import json
import time
import pickle
import re
from tqdm import tqdm
from collections import defaultdict, Counter
from underthesea import word_tokenize

FOLDER_PATH = "Text"
DOC_FILE = "documents.pkl"
FILE_FILE = "filenames.pkl"

def clean_text(text):
    # Loại bỏ các ký tự đặc biệt, giữ lại chữ, số và khoảng trắng
    text = re.sub(r"[^\w\s]", " ", text)
    # Bỏ các ký tự xuống dòng, tab và chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text)
    return text.strip()

start_time = time.time()

# === Kiểm tra và nạp dữ liệu nếu đã tồn tại ===
if os.path.exists(DOC_FILE) and os.path.exists(FILE_FILE):
    print("Tải lại documents và filenames từ file lưu...")
    with open(DOC_FILE, "rb") as f:
        documents = pickle.load(f)
    with open(FILE_FILE, "rb") as f:
        filenames = pickle.load(f)
else:
    print("Đọc và xử lý văn bản từ thư mục...")
    documents = []
    filenames = []
    for file in tqdm(os.listdir(FOLDER_PATH)):
        if file.endswith(".txt"):
            filepath = os.path.join(FOLDER_PATH, file)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                cleaned = clean_text(content)  # Tiền xử lý trước
                tokens = word_tokenize(cleaned, format="text").lower().split()
                documents.append(tokens)
                filenames.append(file)

    # Lưu lại để lần sau dùng
    with open(DOC_FILE, "wb") as f:
        pickle.dump(documents, f)
    with open(FILE_FILE, "wb") as f:
        pickle.dump(filenames, f)

# === Bước 1: Xây chỉ mục ngược (Inverted Index) ===
print("Xây inverted index để tính DF nhanh...")
inverted_index = defaultdict(set)
vocab = set()
for doc_id, doc in enumerate(tqdm(documents)):
    unique_terms = set(doc)
    vocab.update(unique_terms)
    for term in unique_terms:
        inverted_index[term].add(doc_id)

vocab = sorted(vocab)

# === Bước 2: Tính IDF từ inverted index ===
print("Tính IDF cho từng từ...")
N = len(documents)
idf = {term: math.log(N / (1 + len(inverted_index[term]))) for term in vocab}

# === Bước 3: Tính TF-IDF từng văn bản và lưu metadata ===
print("Tính TF-IDF và lưu metadata...")
metadata = []
tfidf_data = []

for doc_id in tqdm(range(N)):
    doc = documents[doc_id]
    filename = filenames[doc_id]
    filepath = os.path.join(FOLDER_PATH, filename)
    word_count = len(doc)
    tf = Counter(doc)

    tfidf_vector = []
    for term in vocab:
        tf_value = tf[term] / word_count if word_count > 0 else 0
        tfidf = tf_value * idf[term]
        tfidf_vector.append(tfidf)

    metadata.append({
        "doc_id": doc_id,
        "filename": filename,
        "word_count": word_count,
        "filepath": filepath
    })

    tfidf_data.append({
        "doc_id": doc_id,
        "vector_sparse": {str(i): tfidf_vector[i] for i in range(len(tfidf_vector)) if tfidf_vector[i] > 0}
    })

# === Bước 4: Ghi metadata ra file ===
print("Ghi metadata vào file metadata.json...")
with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

# === Bước 5: Ghi TF-IDF vector ra file ===
print("Ghi TF-IDF vector vào file tf-idf.json...")
with open("tf-idf.json", "w", encoding="utf-8") as f:
    json.dump(tfidf_data, f, ensure_ascii=False, indent=2)

# === Bước 6: Ghi từ điển vocab vào file vocab.json ===
print("Ghi từ điển vocab vào file vocab.json...")
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

# === Bước 7: Ghi chỉ mục ngược với doc_id ===
print("Ghi chỉ mục ngược vào file inverted_index.json...")
inverted_index_doc_ids = {term: sorted(list(doc_ids)) for term, doc_ids in inverted_index.items()}

with open("inverted_index.json", "w", encoding="utf-8") as f:
    json.dump(inverted_index_doc_ids, f, ensure_ascii=False, indent=2)

elapsed = time.time() - start_time
print(f"Hoàn thành. Thời gian: {elapsed:.2f} giây.")
