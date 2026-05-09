_EMBEDDING_MODEL = None
_EMBEDDINGS_AVAILABLE = False


def get_embedding_model():
    global _EMBEDDING_MODEL, _EMBEDDINGS_AVAILABLE
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            _EMBEDDINGS_AVAILABLE = True
        except Exception as e:
            print(f"[WARN] Failed to load embedding model: {e}")
            _EMBEDDINGS_AVAILABLE = False
    return _EMBEDDING_MODEL


def is_embeddings_available():
    global _EMBEDDINGS_AVAILABLE
    if _EMBEDDING_MODEL is None:
        get_embedding_model()
    return _EMBEDDINGS_AVAILABLE
