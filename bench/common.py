import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))


def load_pairs(path, limit=None):
    """Read question1, question2, is_duplicate (or `label`) from CSV/TSV."""
    path = Path(path)
    delim = "\t" if path.suffix == ".tsv" else ","
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=delim):
            label = r.get("is_duplicate", r.get("label"))
            if r.get("question1") and r.get("question2") and label in ("0", "1"):
                rows.append((r["question1"], r["question2"], int(label)))
    return rows[:limit] if limit else rows
