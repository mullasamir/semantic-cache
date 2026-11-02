from .base import VectorStore


def get_vector_store(settings, db) -> VectorStore:
    if settings.vector_store == "pinecone":
        from .pinecone_store import PineconeStore

        return PineconeStore(settings)
    from .pgvector_store import PgVectorStore

    return PgVectorStore(db)
