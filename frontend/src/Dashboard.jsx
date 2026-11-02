import { useEffect, useRef, useState } from "react";
import { clearCache, getConfig, getLogs, getMetrics, getSavings, setConfig } from "./api.js";
import { ms, pct, usd } from "./format.js";

export default function Dashboard() {
  const [m, setM] = useState(null);
  const [logs, setLogs] = useState([]);
  const [cfg, setCfg] = useState(null);
  const [monthly, setMonthly] = useState(1000000);
  const [sav, setSav] = useState(null);
  const [error, setError] = useState("");
  const timer = useRef(null);

  async function refresh() {
    try {
      const [mm, ll, cc] = await Promise.all([getMetrics(), getLogs(15), getConfig()]);
      setM(mm);
      setLogs(ll);
      setCfg((prev) => prev ?? cc);
      setError("");
    } catch (e) {
      setError(`Could not load metrics. Is the backend running on port 8000? (${e.message})`);
    }
  }

  useEffect(() => {
    refresh();
    timer.current = setInterval(refresh, 3000);
    return () => clearInterval(timer.current);
  }, []);

  useEffect(() => {
    getSavings(monthly || 0).then(setSav).catch(() => {});
  }, [monthly, m?.total]);

  async function applyThreshold(v) {
    setCfg((c) => ({ ...c, similarity_threshold: v }));
    try {
      await setConfig({ similarity_threshold: v });
    } catch (e) {
      setError(e.message);
    }
  }

  async function applyVerify(mode) {
    setCfg((c) => ({ ...c, verify_mode: mode }));
    try {
      setCfg(await setConfig({ verify_mode: mode }));
    } catch (e) {
      setError(e.message);
    }
  }

  if (error && !m) return <p className="error">{error}</p>;
  if (!m || !cfg) return <p className="muted">Loading…</p>;

  const maxLat = Math.max(m.miss_p50_ms || 0, m.hit_p50_ms || 0, 1);

  return (
    <div className="stack">
      {error && <p className="error">{error}</p>}

      <section className="hero">
        <div>
          <div className="big">{pct(m.hit_rate)}</div>
          <div className="muted">of requests answered from cache ({m.hits} of {m.eligible})</div>
        </div>
        <div className="race" aria-label="Median latency, hit versus miss">
          <div className="bar-row">
            <span>Cache hit</span>
            <div className="bar hit" style={{ width: `${Math.max(2, ((m.hit_p50_ms || 0) / maxLat) * 100)}%` }} />
            <b>{ms(m.hit_p50_ms)}</b>
          </div>
          <div className="bar-row">
            <span>LLM call</span>
            <div className="bar miss" style={{ width: `${Math.max(2, ((m.miss_p50_ms || 0) / maxLat) * 100)}%` }} />
            <b>{ms(m.miss_p50_ms)}</b>
          </div>
          <div className="muted small">
            Median latency{m.speedup ? ` · ${m.speedup.toFixed(0)}× faster on a hit` : ""} · p95: hit {ms(m.hit_p95_ms)}, miss {ms(m.miss_p95_ms)}
          </div>
        </div>
      </section>

      <section className="grid">
        <Stat label="LLM spend" value={usd(m.spent_usd)} />
        <Stat label="Saved by cache" value={usd(m.saved_usd)} note={`${pct(m.savings_pct)} less than no cache`} />
        <Stat label="Requests" value={m.total} note={`${m.bypassed} skipped the cache`} />
        <Stat label="Cached answers" value={m.cache_entries} />
      </section>

      <section className="panel stack">
        <h2>Matching rules</h2>
        <label>
          Similarity threshold: <b>{cfg.similarity_threshold.toFixed(2)}</b>
          <input type="range" min="0.5" max="0.99" step="0.01" value={cfg.similarity_threshold}
                 onChange={(e) => applyThreshold(parseFloat(e.target.value))} />
        </label>
        <p className="muted small">
          Lower finds more repeats but risks serving the wrong answer. Higher is safer but misses paraphrases. Run the benchmark to find the best point for your data.
        </p>
        <label className="inline">
          Second-stage check for borderline matches
          <select value={cfg.verify_mode} onChange={(e) => applyVerify(e.target.value)}>
            <option value="none">Off</option>
            <option value="crossencoder">Cross-encoder</option>
            <option value="llm">LLM judge</option>
          </select>
        </label>
        <div>
          <button className="ghost" onClick={async () => { await clearCache(); refresh(); }}>Clear cache</button>
        </div>
      </section>

      <section className="panel stack">
        <h2>Projected savings</h2>
        <label className="inline">
          Requests per month
          <input type="number" min="0" step="10000" value={monthly} onChange={(e) => setMonthly(parseInt(e.target.value || "0", 10))} />
        </label>
        {sav && (
          <div className="grid">
            <Stat label="Without cache" value={usd(sav.monthly_cost_without_cache_usd)} />
            <Stat label="With cache" value={usd(sav.monthly_cost_with_cache_usd)} />
            <Stat label="Monthly savings" value={usd(sav.monthly_savings_usd)} note={`at ${pct(sav.hit_rate)} hit rate`} />
          </div>
        )}
        <p className="muted small">Uses your measured hit rate and average cost per LLM call so far.</p>
      </section>

      <section className="panel">
        <h2>Recent requests</h2>
        {logs.length === 0 ? (
          <p className="muted">Nothing here yet. Send a request from the Playground.</p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Question</th><th>Result</th><th>Similarity</th><th>Latency</th></tr>
              </thead>
              <tbody>
                {logs.map((l) => (
                  <tr key={l.id}>
                    <td className="qcell">{l.query}</td>
                    <td><span className={`badge ${l.cache_hit ? "hit" : l.bypassed ? "bypass" : "miss"}`}>{l.cache_hit ? "Hit" : l.bypassed ? "Skipped" : "Miss"}</span></td>
                    <td>{l.similarity != null ? l.similarity.toFixed(3) : "–"}</td>
                    <td>{ms(l.latency_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function Stat({ label, value, note }) {
  return (
    <div className="stat">
      <div className="stat-v">{value}</div>
      <div className="stat-l">{label}</div>
      {note && <div className="muted small">{note}</div>}
    </div>
  );
}
