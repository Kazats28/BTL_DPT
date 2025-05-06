import time

from .tokenizer import tokenize
from .utils import cosine_similarity_sparse, build_query_vector

def search_query(file_path, vocab, tfidf_data, inverted_index):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    tokens = tokenize(content)
    start1 = time.time()
    query_vector = build_query_vector(tokens, vocab, tfidf_data)
    duration1 = time.time() - start1
    print(f"build_query_vector: {duration1:.2f}")

    start2 = time.time()
    # Lọc các văn bản chứa ít nhất một từ trong truy vấn
    matching_ids = set()
    for term in tokens:
        if term in inverted_index:
            matching_ids.update(map(int, inverted_index[term]))  # danh sách doc_id
    duration2 = time.time() - start2
    print(f"matching_ids: {duration2:.2f}")

    start3 = time.time()
    results = []
    for doc_id in matching_ids:  # chỉ tính cosine với những văn bản này
        if doc_id in tfidf_data:
            doc_vector = tfidf_data[doc_id]
            score = cosine_similarity_sparse(query_vector, doc_vector)
            if score > 0:
                results.append((doc_id, score))
    duration3 = time.time() - start3
    print(f"cosine_similarity_sparse: {duration3:.2f}")

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:3]
