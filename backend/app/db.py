import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
  id BIGSERIAL PRIMARY KEY,
  namespace TEXT NOT NULL,
  query TEXT NOT NULL,
  response TEXT NOT NULL,
  input_tokens INT NOT NULL DEFAULT 0,
  output_tokens INT NOT NULL DEFAULT 0,
  cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
  hits INT NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_entries_ns ON cache_entries(namespace);
CREATE INDEX IF NOT EXISTS idx_entries_exp ON cache_entries(expires_at);

CREATE TABLE IF NOT EXISTS request_logs (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  namespace TEXT NOT NULL,
  query TEXT NOT NULL,
  cache_hit BOOLEAN NOT NULL,
  bypassed BOOLEAN NOT NULL DEFAULT FALSE,
  similarity DOUBLE PRECISION,
  verified BOOLEAN,
  nearest_entry_id BIGINT,
  nearest_query TEXT,
  latency_ms DOUBLE PRECISION NOT NULL,
  cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
  saved_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
  input_tokens INT NOT NULL DEFAULT 0,
  output_tokens INT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_logs_ts ON request_logs(ts);
CREATE INDEX IF NOT EXISTS idx_logs_ns ON request_logs(namespace);
"""

VECTOR_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS cache_vectors (
  entry_id BIGINT PRIMARY KEY REFERENCES cache_entries(id) ON DELETE CASCADE,
  namespace TEXT NOT NULL,
  embedding vector({dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS cache_vectors_ns ON cache_vectors(namespace);
CREATE INDEX IF NOT EXISTS cache_vectors_hnsw ON cache_vectors USING hnsw (embedding vector_cosine_ops);
"""


class DB:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def fetchall(self, sql, params=None) -> list[dict]:
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def fetchone(self, sql, params=None) -> dict | None:
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchone()

    def execute(self, sql, params=None) -> int:
        with self.pool.connection() as conn:
            cur = conn.execute(sql, params)
            return cur.rowcount

    def close(self):
        self.pool.close()


def init_db(settings) -> DB:
    use_pgvector = settings.vector_store == "pgvector"
    try:
        with psycopg.connect(settings.database_url, autocommit=True) as conn:
            conn.execute(SCHEMA)
            if use_pgvector:
                conn.execute(VECTOR_SCHEMA.format(dim=settings.embedding_dim))
    except psycopg.errors.FeatureNotSupported as e:
        raise RuntimeError(
            "pgvector extension is not installed in this Postgres. See README > 'Install pgvector', "
            "or set VECTOR_STORE=pinecone."
        ) from e

    def configure(conn):
        if use_pgvector:
            from pgvector.psycopg import register_vector

            register_vector(conn)
            conn.commit()

    pool = ConnectionPool(settings.database_url, min_size=1, max_size=10, configure=configure, open=True)
    return DB(pool)
