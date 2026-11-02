from .base import VectorStore


class PineconeStore(VectorStore):
    """Pinecone serverless. Note: freshly upserted vectors can take a moment to
    become queryable (eventual consistency), which lowers hit rate on immediate repeats."""

    def __init__(self, settings):
        if not settings.pinecone_api_key:
            raise RuntimeError("VECTOR_STORE=pinecone requires PINECONE_API_KEY")
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=settings.pinecone_api_key)
        if settings.pinecone_index not in pc.list_indexes().names():
            pc.create_index(
                name=settings.pinecone_index,
                dimension=settings.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
        self.index = pc.Index(settings.pinecone_index)

    def add(self, entry_id, namespace, embedding):
        self.index.upsert(vectors=[(str(entry_id), embedding.tolist())], namespace=namespace)

    def search(self, namespace, embedding, top_k=3):
        res = self.index.query(vector=embedding.tolist(), top_k=top_k, namespace=namespace)
        return [(int(m.id), float(m.score)) for m in res.matches]

    def delete(self, entry_ids):
        if entry_ids:
            # Namespace is unknown here; Pinecone deletes by id within a namespace, so scan stats.
            for ns in (self.index.describe_index_stats().namespaces or {}).keys():
                self.index.delete(ids=[str(i) for i in entry_ids], namespace=ns)

    def clear(self, namespace=None):
        if namespace is not None:
            self.index.delete(delete_all=True, namespace=namespace)
        else:
            for ns in (self.index.describe_index_stats().namespaces or {}).keys():
                self.index.delete(delete_all=True, namespace=ns)
