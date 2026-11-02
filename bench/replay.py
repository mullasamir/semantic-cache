"""End-to-end replay against a running API: measured hit rate, p50/p95 latency, cost savings
and correctness (hit on a duplicate = correct, hit on a non-duplicate = false hit).

    python bench/replay.py --limit 48 --threshold 0.85
"""
import argparse
import time

import numpy as np
import requests

from common import ROOT, load_pairs


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--data", default=str(ROOT / "bench" / "sample_pairs.csv"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--verify", choices=["none", "crossencoder", "llm"])
    a = ap.parse_args()

    cfg = {}
    if a.threshold is not None: cfg["similarity_threshold"] = a.threshold
    if a.verify: cfg["verify_mode"] = a.verify
    if cfg: requests.post(f"{a.api}/config", json=cfg).raise_for_status()
    conf = requests.get(f"{a.api}/config").json()

    pairs = load_pairs(a.data, a.limit)
    ns = f"bench-{int(time.time())}"  # isolated namespace per run
    print(f"namespace={ns} pairs={len(pairs)} threshold={conf['similarity_threshold']} verify={conf['verify_mode']}")

    def chat(q):
        return requests.post(f"{a.api}/chat", json={"query": q, "namespace": ns}).json()

    print("phase 1: seeding the cache with each q1 ...")
    for q1, _, _ in pairs:
        chat(q1)

    print("phase 2: replaying paraphrases / look-alikes (q2) ...")
    res = []
    for q1, q2, dup in pairs:
        r = chat(q2)
        res.append((r, q1, dup))

    hits = [r for r, _, _ in res if r["cache_hit"]]
    miss = [r for r, _, _ in res if not r["cache_hit"]]
    correct = sum(1 for r, q1, d in res if r["cache_hit"] and d and r["nearest_query"] == q1)
    false_hits = sum(1 for r, _, d in res if r["cache_hit"] and not d)
    dups = sum(1 for _, _, d in res if d)
    saved = sum(r["saved_usd"] for r, _, _ in res)
    spent = sum(r["cost_usd"] for r, _, _ in res)

    print("\n=== results (phase 2) ===")
    print(f"requests           : {len(res)}")
    print(f"hit rate           : {len(hits)/len(res):.1%}")
    print(f"duplicates caught  : {correct}/{dups} ({correct/max(1,dups):.1%})")
    print(f"false-hit rate     : {false_hits/max(1,len(hits)):.1%} of hits ({false_hits} wrong answers)")
    print(f"hit latency        : p50 {pct([r['latency_ms'] for r in hits],50):.1f} ms | p95 {pct([r['latency_ms'] for r in hits],95):.1f} ms")
    print(f"miss latency       : p50 {pct([r['latency_ms'] for r in miss],50):.1f} ms | p95 {pct([r['latency_ms'] for r in miss],95):.1f} ms")
    print(f"LLM spend          : ${spent:.5f}   saved: ${saved:.5f}   ({saved/max(1e-12,spent+saved):.1%} cost reduction)")


if __name__ == "__main__":
    main()
