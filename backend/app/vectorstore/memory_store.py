import numpy as np

from .base import VectorStore


class InMemoryVectorStore(VectorStore):
    """Brute-force store used by tests and offline experiments."""

    def __init__(self):
        self.items: dict[int, tuple[str, np.ndarray]] = {}

    def add(self, entry_id, namespace, embedding):
        self.items[entry_id] = (namespace, np.asarray(embedding, dtype=np.float32))

    def search(self, namespace, embedding, top_k=3):
        scored = [(i, float(np.dot(v, embedding))) for i, (ns, v) in self.items.items() if ns == namespace]
        return sorted(scored, key=lambda x: -x[1])[:top_k]

    def delete(self, entry_ids):
        for i in entry_ids:
            self.items.pop(i, None)

    def clear(self, namespace=None):
        self.items = {i: x for i, x in self.items.items() if namespace is not None and x[0] != namespace}
