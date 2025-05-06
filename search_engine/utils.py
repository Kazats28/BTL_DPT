import math
import time
from collections import Counter

def cosine_similarity_sparse(vec1, vec2):
    dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in set(vec1) | set(vec2))
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

def build_query_vector(tokens, vocab, tfidf_data):
    vocab_index = {term: idx for idx, term in enumerate(vocab)}
    word_count = len(tokens)
    tf = Counter(tokens)

    start2 = time.time()
    idf = {}
    for term in vocab:
        idx = str(vocab_index[term])
        df = sum(1 for doc in tfidf_data.values() if idx in doc)
        idf[term] = math.log(len(tfidf_data) / (1 + df))
    duration2 = time.time() - start2
    print(f"idf: {duration2:.2f}")

    query_vector = {}
    for term in set(tokens):
        if term in vocab_index:
            idx = vocab_index[term]
            tf_value = tf[term] / word_count
            tfidf = tf_value * idf[term]
            if tfidf > 0:
                query_vector[str(idx)] = tfidf
    return query_vector

