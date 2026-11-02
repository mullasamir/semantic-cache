"""Download a sample of the Quora Question Pairs dataset (GLUE/QQP) for realistic numbers.

    pip install datasets
    python bench/download_qqp.py --n 2000
"""
import argparse
import csv
import random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

from datasets import load_dataset  # noqa: E402

ds = load_dataset("nyu-mll/glue", "qqp", split="validation")
idx = list(range(len(ds)))
random.Random(args.seed).shuffle(idx)
out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)
with open(out / "qqp_sample.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["question1", "question2", "is_duplicate"])
    for i in idx[: args.n]:
        r = ds[i]
        w.writerow([r["question1"], r["question2"], r["label"]])
print(f"wrote {out / 'qqp_sample.csv'}")
