import numpy as np

from app.embeddings import HashEmbedder
from app.vectorstore.memory_store import InMemoryVectorStore


def test_embeddings_are_normalised_and_similar_text_is_closer():
    e = HashEmbedder(384)
    v = e.encode(["how do i reset my password", "how can i reset my password", "best pizza in rome"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


def test_namespaces_are_isolated():
    e = HashEmbedder(384)
    v = e.encode(["reset password"])[0]
    s = InMemoryVectorStore()
    s.add(1, "alice", v)
    assert s.search("alice", v)[0][0] == 1
    assert s.search("bob", v) == []


def test_delete_and_clear():
    e = HashEmbedder(384)
    v = e.encode(["x y z"])[0]
    s = InMemoryVectorStore()
    s.add(1, "a", v); s.add(2, "b", v)
    s.delete([1])
    assert s.search("a", v) == []
    s.clear("b")
    assert s.search("b", v) == []
