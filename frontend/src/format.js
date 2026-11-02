export const ms = (v) => (v == null ? "–" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(0)} ms`);
export const pct = (v) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);
export const usd = (v) =>
  v == null ? "–" : v !== 0 && Math.abs(v) < 0.01 ? `$${v.toFixed(5)}` : `$${v.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
