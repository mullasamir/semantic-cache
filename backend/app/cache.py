import logging
import time
from datetime import datetime, timedelta, timezone

from .policy import decide, is_time_sensitive

log = logging.getLogger("semantic_cache")


class SemanticCache:
    def __init__(self, settings, db, embedder, store, llm, verifier):
        self.s, self.db, self.embedder, self.store, self.llm, self.verifier = settings, db, embedder, store, llm, verifier

    # ------------------------------------------------------------------ main flow
    def handle(self, query: str, namespace: str = "global", bypass: bool = False) -> dict:
        t0 = time.perf_counter()
        query = query.strip()
        bypassed = bypass or is_time_sensitive(query, self.s.time_sensitive_regex)

        if bypassed:  # never read or write the cache
            res = self.llm.generate(query)
            return self._finish(t0, namespace, query, res.text, hit=False, bypassed=True,
                                cost=res.cost_usd, tin=res.input_tokens, tout=res.output_tokens)

        emb = self.embedder.encode([query])[0]
        entry, sim = None, None
        for entry_id, score in self.store.search(namespace, emb, top_k=3):
            row = self.db.fetchone(
                "SELECT id, query, response, cost_usd FROM cache_entries "
                "WHERE id = %s AND (expires_at IS NULL OR expires_at > now())", (entry_id,))
            if row:
                entry, sim = row, score
                break

        action, verified = "miss", None
        if entry:
            action = decide(sim, self.s.similarity_threshold, self.s.verify_mode, self.s.verify_low, self.s.verify_high)
            if action == "verify":
                verified = self.verifier.approve(query, entry["query"])
                action = "hit" if verified else "miss"

        if action == "hit":
            self.db.execute("UPDATE cache_entries SET hits = hits + 1 WHERE id = %s", (entry["id"],))
            return self._finish(t0, namespace, query, entry["response"], hit=True, bypassed=False,
                                sim=sim, verified=verified, entry=entry, saved=entry["cost_usd"])

        res = self.llm.generate(query)
        self._store(namespace, query, emb, res)
        return self._finish(t0, namespace, query, res.text, hit=False, bypassed=False, sim=sim,
                            verified=verified, entry=entry, cost=res.cost_usd,
                            tin=res.input_tokens, tout=res.output_tokens)

    # ------------------------------------------------------------------ helpers
    def _store(self, namespace, query, emb, res):
        ttl = self.s.ttl_seconds
        expires = datetime.now(timezone.utc) + timedelta(seconds=ttl) if ttl > 0 else None
        row = self.db.fetchone(
            "INSERT INTO cache_entries (namespace, query, response, input_tokens, output_tokens, cost_usd, expires_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (namespace, query, res.text, res.input_tokens, res.output_tokens, res.cost_usd, expires))
        self.store.add(row["id"], namespace, emb)

    def _finish(self, t0, namespace, query, answer, *, hit, bypassed, sim=None, verified=None,
                entry=None, cost=0.0, saved=0.0, tin=0, tout=0) -> dict:
        latency_ms = (time.perf_counter() - t0) * 1000
        nearest_id = entry["id"] if entry else None
        nearest_q = entry["query"] if entry else None
        try:
            self.db.execute(
                "INSERT INTO request_logs (namespace, query, cache_hit, bypassed, similarity, verified, "
                "nearest_entry_id, nearest_query, latency_ms, cost_usd, saved_usd, input_tokens, output_tokens) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (namespace, query, hit, bypassed, sim, verified, nearest_id, nearest_q,
                 latency_ms, cost, saved, tin, tout))
        except Exception:  # logging must never break a request
            log.exception("failed to write request log")
        return {
            "answer": answer, "cache_hit": hit, "bypassed": bypassed, "similarity": sim,
            "nearest_query": nearest_q, "verified": verified, "latency_ms": round(latency_ms, 2),
            "cost_usd": cost, "saved_usd": saved, "namespace": namespace,
        }

    def purge_expired(self) -> int:
        rows = self.db.fetchall("DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < now() RETURNING id")
        ids = [r["id"] for r in rows]
        if ids:
            self.store.delete(ids)
        return len(ids)

    def clear(self, namespace: str | None = None) -> None:
        if namespace is None:
            self.db.execute("DELETE FROM cache_entries")
        else:
            self.db.execute("DELETE FROM cache_entries WHERE namespace = %s", (namespace,))
        self.store.clear(namespace)
