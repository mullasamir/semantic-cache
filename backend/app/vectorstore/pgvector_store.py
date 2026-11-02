from .base import VectorStore


class PgVectorStore(VectorStore):
    def __init__(self, db):
        self.db = db

    def add(self, entry_id, namespace, embedding):
        self.db.execute(
            "INSERT INTO cache_vectors (entry_id, namespace, embedding) VALUES (%s, %s, %s) "
            "ON CONFLICT (entry_id) DO UPDATE SET embedding = EXCLUDED.embedding",
            (entry_id, namespace, embedding),
        )

    def search(self, namespace, embedding, top_k=3):
        rows = self.db.fetchall(
            "SELECT entry_id, 1 - (embedding <=> %s) AS similarity FROM cache_vectors "
            "WHERE namespace = %s ORDER BY embedding <=> %s LIMIT %s",
            (embedding, namespace, embedding, top_k),
        )
        return [(int(r["entry_id"]), float(r["similarity"])) for r in rows]

    def delete(self, entry_ids):
        if entry_ids:
            self.db.execute("DELETE FROM cache_vectors WHERE entry_id = ANY(%s)", (list(entry_ids),))

    def clear(self, namespace=None):
        if namespace is None:
            self.db.execute("DELETE FROM cache_vectors")
        else:
            self.db.execute("DELETE FROM cache_vectors WHERE namespace = %s", (namespace,))
