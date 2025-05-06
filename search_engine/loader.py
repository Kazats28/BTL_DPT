import json

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_all_data():
    vocab = load_json("vocab.json")
    metadata = load_json("metadata.json")
    tfidf_data = load_json("tf-idf.json")
    tfidf_dict = {doc["doc_id"]: doc["vector_sparse"] for doc in tfidf_data}
    tfidf_data = tfidf_dict
    inverted_index = load_json("inverted_index.json")
    return vocab, metadata, tfidf_data, inverted_index
