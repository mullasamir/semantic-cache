import { useState } from "react";
import { chat } from "./api.js";
import { ms, usd } from "./format.js";

const EXAMPLES = [
  "How do I reset my password?",
  "I forgot my password, how can I change it?",
  "How do I reset my router?",
  "What's the weather today in Paris?",
];

export default function Playground() {
  const [query, setQuery] = useState(EXAMPLES[0]);
  const [namespace, setNamespace] = useState("demo");
  const [bypass, setBypass] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);

  async function send(e) {
    e?.preventDefault();
    if (!query.trim() || busy) return;
    setBusy(true);
    setError("");
    try {
      const r = await chat({ query, namespace, bypass_cache: bypass });
      setHistory((h) => [{ q: query, ...r, id: Date.now() }, ...h]);
    } catch (err) {
      setError(`Could not reach the API. Is the backend running on port 8000? (${err.message})`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      <p className="lede">
        Ask a question, then ask it again in different words. The second answer should come back from the cache in milliseconds.
      </p>

      <form onSubmit={send} className="panel stack">
        <label>
          Question
          <textarea rows={3} value={query} onChange={(e) => setQuery(e.target.value)} />
        </label>
        <div className="chips">
          {EXAMPLES.map((x) => (
            <button type="button" key={x} className="chip" onClick={() => setQuery(x)}>
              {x}
            </button>
          ))}
        </div>
        <div className="row">
          <label className="inline">
            Namespace
            <input value={namespace} onChange={(e) => setNamespace(e.target.value)} />
          </label>
          <label className="check">
            <input type="checkbox" checked={bypass} onChange={(e) => setBypass(e.target.checked)} />
            Skip the cache
          </label>
          <button className="primary" disabled={busy}>
            {busy ? "Asking…" : "Ask"}
          </button>
        </div>
        {error && <p className="error">{error}</p>}
      </form>

      {history.length === 0 && <p className="muted">No requests yet. Press Ask to send the first one.</p>}
      {history.map((h) => (
        <article key={h.id} className={`result ${h.cache_hit ? "hit" : h.bypassed ? "bypass" : "miss"}`}>
          <div className="result-head">
            <strong>{h.cache_hit ? "Cache hit" : h.bypassed ? "Cache skipped" : "Cache miss"}</strong>
            <span className="latency">{ms(h.latency_ms)}</span>
          </div>
          <p className="q">{h.q}</p>
          <p className="a">{h.answer}</p>
          <dl className="facts">
            {h.similarity != null && (
              <div>
                <dt>Similarity</dt>
                <dd>{h.similarity.toFixed(3)}</dd>
              </div>
            )}
            {h.nearest_query && (
              <div>
                <dt>Closest cached question</dt>
                <dd>{h.nearest_query}</dd>
              </div>
            )}
            {h.verified != null && (
              <div>
                <dt>Second-stage check</dt>
                <dd>{h.verified ? "Approved" : "Rejected"}</dd>
              </div>
            )}
            <div>
              <dt>{h.cache_hit ? "Saved" : "Cost"}</dt>
              <dd>{usd(h.cache_hit ? h.saved_usd : h.cost_usd)}</dd>
            </div>
          </dl>
        </article>
      ))}
    </div>
  );
}
