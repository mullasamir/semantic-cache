"""Threshold sweep: hit rate vs false-hit rate on labelled question pairs.

Needs no database or API server. Each pair is treated as "q1 is already cached, q2 arrives":
  - duplicate pair + similarity >= threshold  -> correct hit
  - non-duplicate pair + similarity >= threshold -> FALSE hit (wrong answer served)

    python bench/sweep.py                                   # built-in sample
    python bench/sweep.py --data bench/data/qqp_sample.csv  # realistic numbers
    python bench/sweep.py --data ... --cross-encoder        # also evaluate two-stage
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from common import ROOT, load_pairs

from app.config import settings  # noqa: E402
from app.embeddings import get_embedder  # noqa: E402


def metrics(hit, dup):
    n_hits = int(hit.sum())
    true_hits = int((hit & dup).sum())
    false_hits = int((hit & ~dup).sum())
    return {
        "hit_rate": n_hits / len(hit),                                  # share of requests served from cache
        "recall": true_hits / max(1, int(dup.sum())),                   # share of real duplicates caught
        "false_hit_rate": false_hits / n_hits if n_hits else 0.0,       # share of hits that are wrong
        "precision": true_hits / n_hits if n_hits else 1.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "bench" / "sample_pairs.csv"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--min", type=float, default=0.70)
    ap.add_argument("--max", type=float, default=0.99)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--cross-encoder", action="store_true")
    ap.add_argument("--verify-low", type=float, default=settings.verify_low)
    ap.add_argument("--verify-high", type=float, default=settings.verify_high)
    ap.add_argument("--out", default=str(ROOT / "bench" / "results"))
    a = ap.parse_args()

    pairs = load_pairs(a.data, a.limit)
    dup = np.array([p[2] for p in pairs], dtype=bool)
    print(f"{len(pairs)} pairs ({int(dup.sum())} duplicates), embeddings={settings.embedding_backend}")

    emb = get_embedder(settings)
    sims = (emb.encode([p[0] for p in pairs]) * emb.encode([p[1] for p in pairs])).sum(axis=1)

    rows = []
    print(f"\n{'threshold':>9} {'hit rate':>9} {'recall':>8} {'false-hit':>10}")
    for t in np.arange(a.min, a.max + 1e-9, a.step):
        m = metrics(sims >= t, dup)
        rows.append({"threshold": round(float(t), 3), **m})
        print(f"{t:9.2f} {m['hit_rate']:9.1%} {m['recall']:8.1%} {m['false_hit_rate']:10.1%}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    two_stage = None
    if a.cross_encoder:
        from sentence_transformers import CrossEncoder

        ce = CrossEncoder(settings.cross_encoder_model)
        band = np.where(sims >= a.verify_low)[0]
        scores = np.zeros(len(pairs))
        if len(band):
            scores[band] = ce.predict([(pairs[i][0], pairs[i][1]) for i in band])
        hit = (sims >= a.verify_high) | ((sims >= a.verify_low) & (scores >= settings.cross_encoder_threshold))
        two_stage = metrics(hit, dup)
        single = metrics(sims >= settings.similarity_threshold, dup)
        print(f"\nsingle threshold {settings.similarity_threshold:.2f}: {single}")
        print(f"two-stage [{a.verify_low:.2f}, {a.verify_high:.2f}) + cross-encoder: {two_stage}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = [r["threshold"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, [r["hit_rate"] * 100 for r in rows], label="Hit rate (all requests)", lw=2)
        ax.plot(x, [r["recall"] * 100 for r in rows], label="Duplicates caught", lw=2)
        ax.plot(x, [r["false_hit_rate"] * 100 for r in rows], label="False-hit rate (wrong answers served)", lw=2, color="crimson")
        ax.axvline(settings.similarity_threshold, ls="--", color="gray", lw=1)
        ax.set_xlabel("Similarity threshold"); ax.set_ylabel("%"); ax.grid(alpha=0.3); ax.legend()
        ax.set_title("Threshold sweep: hit rate vs false-hit rate")
        fig.tight_layout(); fig.savefig(out / "sweep.png", dpi=150)
        print(f"\nsaved {out / 'sweep.csv'} and {out / 'sweep.png'}")
    except ImportError:
        print(f"\nsaved {out / 'sweep.csv'} (pip install matplotlib for the chart)")


if __name__ == "__main__":
    main()
