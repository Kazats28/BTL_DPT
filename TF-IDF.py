import os
import math
import json
import time
from tqdm import tqdm
from collections import defaultdict, Counter
from underthesea import word_tokenize

FOLDER_PATH = "Text"
OUTPUT_FILE = "metadata.json"

start_time = time.time()
documents = []
filenames = []

print("Đọc và xử lý văn bản...")
for file in tqdm(os.listdir(FOLDER_PATH)):
    if file.endswith(".txt"):
        filepath = os.path.join(FOLDER_PATH, file)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            tokens = word_tokenize(content, format="text").lower().split()
            documents.append(tokens)
            filenames.append(file)

# === Bước 1: Xây chỉ mục ngược (Inverted Index) ===
print("Xây inverted index để tính DF nhanh...")
inverted_index = defaultdict(set)
vocab = set()
for doc_id, doc in enumerate(tqdm(documents)):
    unique_terms = set(doc)
    vocab.update(unique_terms)
    for term in unique_terms:
        inverted_index[term].add(doc_id)

vocab = sorted(vocab)  # Sắp xếp từ vựng để vector đồng nhất

# === Bước 2: Tính IDF từ inverted index ===
print("Tính IDF cho từng từ...")
N = len(documents)
idf = {term: math.log(N / (1 + len(inverted_index[term]))) for term in vocab}

# === Bước 3: Tính TF-IDF từng văn bản ===
print("Tính TF-IDF cho từng văn bản...")
metadata = []

for i in tqdm(range(N)):
    doc = documents[i]
    filename = filenames[i]
    word_count = len(doc)
    tf = Counter(doc)

    tfidf_vector = []
    for term in vocab:
        tf_value = tf[term] / word_count if word_count > 0 else 0
        tfidf = tf_value * idf[term]
        tfidf_vector.append(tfidf)

    metadata.append({
        "filename": filename,
        "word_count": word_count,
        "vector_sparse": {str(i): tfidf_vector[i] for i in range(len(tfidf_vector)) if tfidf_vector[i] > 0}
    })

# === Bước 4: Ghi kết quả ra JSON ===
print("Ghi metadata vào file JSON...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

elapsed = time.time() - start_time
print(f"Hoàn thành. Thời gian: {elapsed:.2f} giây.")
