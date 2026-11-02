import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .cache import SemanticCache
from .config import settings
from .db import init_db
from .embeddings import get_embedder
from .llm import get_llm
from .metrics import get_logs, get_metrics, project_savings
from .vectorstore import get_vector_store
from .verifier import Verifier

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("semantic_cache")
ctx: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = init_db(settings)
    embedder = get_embedder(settings)
    embedder.encode(["warm up"])
    llm = get_llm(settings)
    cache = SemanticCache(settings, db, embedder, get_vector_store(settings, db), llm, Verifier(settings, llm))
    ctx.update(db=db, cache=cache)
    stop = threading.Event()

    def purge_loop():
        while not stop.wait(60):
            try:
                n = cache.purge_expired()
                if n:
                    log.info("purged %d expired entries", n)
            except Exception:
                log.exception("purge failed")

    threading.Thread(target=purge_loop, daemon=True).start()
    log.info("ready: vector_store=%s embeddings=%s llm=%s/%s", settings.vector_store,
             settings.embedding_backend, settings.llm_provider, settings.llm_model)
    yield
    stop.set()
    db.close()


app = FastAPI(title="Semantic Cache for LLM APIs", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
                   allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    namespace: str = Field(default="global", max_length=100)  # per-user / per-tenant isolation
    bypass_cache: bool = False


class ConfigUpdate(BaseModel):
    similarity_threshold: float | None = Field(default=None, ge=0, le=1)
    verify_mode: str | None = None
    verify_low: float | None = Field(default=None, ge=0, le=1)
    verify_high: float | None = Field(default=None, ge=0, le=1)
    ttl_seconds: int | None = Field(default=None, ge=0)


def _config() -> dict:
    return {
        "similarity_threshold": settings.similarity_threshold, "verify_mode": settings.verify_mode,
        "verify_low": settings.verify_low, "verify_high": settings.verify_high, "ttl_seconds": settings.ttl_seconds,
        "vector_store": settings.vector_store, "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.embedding_model, "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    return ctx["cache"].handle(req.query, req.namespace, req.bypass_cache)


@app.get("/metrics")
def metrics(namespace: str | None = None, window_minutes: int = Query(525600, ge=1)):
    return get_metrics(ctx["db"], namespace, window_minutes)


@app.get("/logs")
def logs(limit: int = Query(50, ge=1, le=500), namespace: str | None = None):
    return get_logs(ctx["db"], limit, namespace)


@app.get("/savings")
def savings(monthly_requests: int = Query(1_000_000, ge=0), hit_rate: float | None = Query(None, ge=0, le=1),
            namespace: str | None = None):
    return project_savings(get_metrics(ctx["db"], namespace), monthly_requests, hit_rate)


@app.get("/config")
def get_config():
    return _config()


@app.post("/config")
def update_config(body: ConfigUpdate):
    data = body.model_dump(exclude_none=True)
    if "verify_mode" in data and data["verify_mode"] not in ("none", "crossencoder", "llm"):
        raise HTTPException(422, "verify_mode must be none, crossencoder or llm")
    for k, v in data.items():
        setattr(settings, k, v)
    return _config()


@app.delete("/cache")
def clear_cache(namespace: str | None = None):
    ctx["cache"].clear(namespace)
    return {"cleared": namespace or "all"}
