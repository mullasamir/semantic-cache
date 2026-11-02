_AGG = """
SELECT
  count(*)                                              AS total,
  count(*) FILTER (WHERE bypassed)                      AS bypassed,
  count(*) FILTER (WHERE cache_hit)                     AS hits,
  coalesce(sum(cost_usd), 0)                            AS spent_usd,
  coalesce(sum(saved_usd), 0)                           AS saved_usd,
  percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE cache_hit)                      AS hit_p50_ms,
  percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE cache_hit)                      AS hit_p95_ms,
  percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE NOT cache_hit AND NOT bypassed) AS miss_p50_ms,
  percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE NOT cache_hit AND NOT bypassed) AS miss_p95_ms,
  avg(cost_usd) FILTER (WHERE NOT cache_hit AND NOT bypassed)                                            AS avg_miss_cost_usd
FROM request_logs
WHERE ts > now() - make_interval(mins => %s)
  AND (%s::text IS NULL OR namespace = %s)
"""


def get_metrics(db, namespace=None, window_minutes=525600) -> dict:
    r = db.fetchone(_AGG, (window_minutes, namespace, namespace))
    eligible = r["total"] - r["bypassed"]
    out = {k: (float(v) if v is not None and not isinstance(v, int) else v) for k, v in r.items()}
    out["eligible"] = eligible
    out["hit_rate"] = (r["hits"] / eligible) if eligible else 0.0
    out["speedup"] = (out["miss_p50_ms"] / out["hit_p50_ms"]) if out.get("hit_p50_ms") and out.get("miss_p50_ms") else None
    out["cost_without_cache_usd"] = out["spent_usd"] + out["saved_usd"]
    out["savings_pct"] = (out["saved_usd"] / out["cost_without_cache_usd"]) if out["cost_without_cache_usd"] else 0.0
    out["cache_entries"] = db.fetchone("SELECT count(*) AS n FROM cache_entries")["n"]
    return out


def get_logs(db, limit=50, namespace=None) -> list[dict]:
    return db.fetchall(
        "SELECT id, ts, namespace, query, cache_hit, bypassed, similarity, verified, nearest_query, "
        "latency_ms, cost_usd, saved_usd FROM request_logs "
        "WHERE (%s::text IS NULL OR namespace = %s) ORDER BY id DESC LIMIT %s",
        (namespace, namespace, limit))


def project_savings(m: dict, monthly_requests: int, hit_rate: float | None = None) -> dict:
    hr = m["hit_rate"] if hit_rate is None else hit_rate
    avg_cost = m.get("avg_miss_cost_usd") or 0.0
    baseline = monthly_requests * avg_cost
    saved = baseline * hr
    return {"monthly_requests": monthly_requests, "hit_rate": hr, "avg_llm_cost_usd": avg_cost,
            "monthly_cost_without_cache_usd": baseline, "monthly_cost_with_cache_usd": baseline - saved,
            "monthly_savings_usd": saved}
