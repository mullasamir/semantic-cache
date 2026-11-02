import hashlib
import re

import numpy as np


class Embedder:
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # returns L2-normalised float32 (n, dim)
        raise NotImplementedError


class HashEmbedder(Embedder):
    """Dependency-free bag-of-words hashing embedder. Good enough to demo the
    pipeline offline and for tests; it has NO real semantic understanding."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = re.findall(r"[a-z0-9]+", text.lower())
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for f in feats:
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def get_embedder(settings) -> Embedder:
    if settings.embedding_backend == "hash":
        emb: Embedder = HashEmbedder(settings.embedding_dim)
    else:
        emb = SentenceTransformerEmbedder(settings.embedding_model)
    if emb.dim != settings.embedding_dim:
        raise RuntimeError(
            f"Embedding model produces {emb.dim}-dim vectors but EMBEDDING_DIM={settings.embedding_dim}. "
            "Update EMBEDDING_DIM (and recreate the cache_vectors table / Pinecone index)."
        )
    return emb
