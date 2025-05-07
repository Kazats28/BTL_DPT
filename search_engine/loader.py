import pickle

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_all_data():
    vocab = load_pkl("vocab.pkl")
    metadata = load_pkl("metadata.pkl")
    tfidf_data = load_pkl("tf-idf.pkl")
    tfidf_dict = {doc["doc_id"]: doc["vector_sparse"] for doc in tfidf_data}
    tfidf_data = tfidf_dict
    inverted_index = load_pkl("inverted_index.pkl")
    norm2 = load_pkl("norm2.pkl")
    idf = load_pkl("idf.pkl")
    return vocab, metadata, tfidf_data, inverted_index, norm2, idf
