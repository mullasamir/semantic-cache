# semantic-cache

A small caching layer that sits in front of an LLM API and answers repeat questions without calling the model again, even when the repeat is worded differently.

I built it to find out how much a semantic cache actually saves, and how often it gets things wrong. Most write-ups on this topic show the savings and skip the errors. The part I cared about was the second one.

![Threshold sweep](docs/sweep.png)

## The problem

"How do I reset my password?" and "I forgot my password, how can I change it?" are the same question, but an ordinary cache sees two different strings and pays for two LLM calls. A semantic cache compares meaning instead of text. The risk is that it also treats "how do I cancel my subscription?" and "how do I renew my subscription?" as the same question, and then serves a confidently wrong answer.

So the interesting question isn't whether it can save money. It's how much you can save before the wrong answers start to matter.

## How it works

1. The incoming question is turned into a 384-dimension vector with `all-MiniLM-L6-v2`.
2. pgvector finds the closest previously answered question in that user's namespace.
3. A very high similarity (default 0.97 and above) is a hit. A very low one (below 0.80) is a miss.
4. Anything in between goes to a cross-encoder, which reads both questions together and decides whether they mean the same thing.
5. On a miss, the LLM is called and the answer is stored with a TTL. Every request, hit or miss, is logged to Postgres with its similarity score, latency and cost.

Questions that go stale quickly ("weather today", "latest news") skip the cache entirely, and each user or tenant gets a separate namespace so answers are never shared across them.

**Stack:** FastAPI, PostgreSQL with pgvector, sentence-transformers, React (Vite). Pinecone is supported as an alternative vector store behind the same small interface.

## What I found

Evaluated on 2,000 pairs from Quora Question Pairs (706 of them true duplicates). Each pair simulates "the first question is already cached, the second one arrives." A hit on a non-duplicate counts as a wrong answer.

| Setup | Hit rate | Duplicates caught | Wrong answers, share of hits |
|---|---|---|---|
| Single threshold 0.90 | 17.9% | 39.8% | 21.7% |
| Single threshold 0.83 | 33.6% | 67.6% | 29.0% |
| Cross-encoder, auto-accept at 0.93 | 28.2% | 70.0% | 12.3% |
| Cross-encoder, auto-accept at 0.97 | 26.5% | 68.8% | 8.3% |
| Cross-encoder, auto-accept at 0.99 | 26.3% | 68.6% | 7.8% |

Comparing at roughly equal recall (single threshold at 0.83 against the 0.97 setup), the cross-encoder cuts wrong answers from 29.0% to 8.3% of hits, a drop of about 70%. It does this while giving up very little: the share of requests answered correctly from cache stays near 24 to 25 percent either way. The single threshold's extra hits are mostly wrong ones.

Three things surprised me:

- **A similarity score alone can't reach a safe error rate.** Even at 0.97 and above, 10% of hits were wrong. Some different questions really do land very close together in embedding space.
- **The auto-accept line mattered more than the verifier's own threshold.** Changing the cross-encoder's cutoff from 0.5 to 0.7 barely moved anything. Raising the auto-accept line from 0.93 to 0.97 removed about a third of the remaining errors, because the highest-scoring matches were never being double-checked.
- **Returns flatten out.** Going from 0.97 to 0.99 improved things by half a point and sends more traffic through the slower model, so I stopped at 0.97.

### End-to-end replay

The same idea through the real API, 200 pairs, with a simulated LLM:

| | Single threshold 0.83 | Cross-encoder |
|---|---|---|
| Hit rate | 35.0% | 26.0% |
| Duplicates caught | 47 of 68 | 44 of 68 |
| Wrong answers served | 22 | 7 |
| Hit latency, p50 | 23 ms | 52 to 70 ms |
| Miss latency, p50 | about 850 ms (simulated) | about 850 ms (simulated) |

A cache hit is still more than ten times faster than a miss, even with the cross-encoder running. The verifier costs roughly 30 to 45 ms per hit on my CPU, which is the price of the lower error rate. The hit latency moved between 52 and 73 ms across repeat runs on a laptop, so treat it as a range.

On cost, the replay reported a 25% reduction, but that counts the 7 wrong answers as savings. Counting only correct hits, it's closer to 22%. The percentage doesn't depend much on the price you assume, since every call costs roughly the same. The dollar figures do, so the project reads prices from `.env` rather than hardcoding any.

## Limits of these numbers

- The cross-encoder (`quora-distilroberta-base`) was trained on Quora-style data, so results on QQP are probably better than what you'd see on other domains.
- QQP labels are noisy. Some "non-duplicates" are arguably the same question, so a few of the wrong answers above aren't really wrong.
- 35% of the pairs are duplicates, which is much higher than most real traffic. Absolute hit rates will be lower in practice. The comparison between setups is the part to trust.
- The LLM in the benchmarks is a mock with a simulated delay of around 800 ms and an assumed output of roughly 100 tokens. Matching accuracy doesn't depend on the LLM at all, but the latency and dollar figures do. Cost used $0.30 input and $2.50 output per 1M tokens, set in `.env`.
- Wrong-answer rates need labeled data. The live dashboard can't tell you how often it's wrong, only the benchmark can.

## Running it

You need Python 3.10+, Node 18+, and Postgres with the pgvector extension. The easiest way to get the database is Docker:

```bash
docker run -d --name pgvec -p 5433:5432 \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=semantic_cache \
  pgvector/pgvector:pg16
```

Then:

```bash
cp .env.example .env        # set DATABASE_URL, e.g. postgresql://postgres:postgres@localhost:5433/semantic_cache

cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The UI is at `http://localhost:5173` and the API docs at `http://localhost:8000/docs`. The first start downloads the embedding model (about 90 MB). By default the LLM is a mock, so you don't need an API key. To use a real model, set `LLM_PROVIDER` and the matching key in `.env`, and install `backend/requirements-optional.txt`.

For the verification stage, set these in `.env`:

```
VERIFY_MODE=crossencoder
VERIFY_LOW=0.80
VERIFY_HIGH=0.97
```

## Reproducing the benchmarks

```bash
pip install -r bench/requirements.txt datasets
python bench/download_qqp.py --n 2000
python bench/sweep.py --data bench/data/qqp_sample.csv --cross-encoder --verify-high 0.97
python bench/replay.py --data bench/data/qqp_sample.csv --limit 200 --verify crossencoder
```

`sweep.py` needs no server and writes a CSV and the chart. `replay.py` needs the backend running and uses a fresh namespace each time. Run the replay twice and keep the second result, because the first includes model loading. The built-in `bench/sample_pairs.csv` has only 48 pairs and is just for checking that things work.

## API

| Method | Path | What it does |
|---|---|---|
| POST | `/chat` | `{query, namespace?, bypass_cache?}`, returns the answer plus hit/miss, similarity, latency and cost |
| GET | `/metrics` | Hit rate, p50 and p95 latency for hits and misses, spend and savings |
| GET | `/logs` | Recent requests |
| GET | `/savings` | Monthly projection from your measured hit rate |
| GET, POST | `/config` | Read or change the threshold and verify mode at runtime |
| DELETE | `/cache` | Clear cached entries |

## Layout

```
backend/app/     API, cache flow, decision logic, embeddings, LLM providers, vector stores
backend/tests/   unit tests (cd backend && pytest -q tests)
bench/           threshold sweep, end-to-end replay, QQP download script
frontend/src/    playground and dashboard
```

A `docker-compose.yml` is included for running everything together with `docker compose up --build`.

## What I'd do next

Compare pgvector against Pinecone on lookup latency, run the benchmarks against a real model, and test the verifier on a domain other than Quora. A Redis layer for exact-match repeats would also make sense in front of all this.
