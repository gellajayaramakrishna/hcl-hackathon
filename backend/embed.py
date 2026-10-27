# Step 3: turn text into a 384-number vector with bge-small (CPU). The model loads once.
from sentence_transformers import SentenceTransformer

from backend import config

_model = None


def embed(text):
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
    return _model.encode(text, normalize_embeddings=True).tolist()
