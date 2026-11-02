const BASE = import.meta.env.VITE_API_URL || "/api";

async function req(path, opts = {}) {
  const res = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

export const chat = (body) => req("/chat", { method: "POST", body });
export const getMetrics = (namespace) =>
  req("/metrics" + (namespace ? `?namespace=${encodeURIComponent(namespace)}` : ""));
export const getLogs = (limit = 20) => req(`/logs?limit=${limit}`);
export const getConfig = () => req("/config");
export const setConfig = (body) => req("/config", { method: "POST", body });
export const getSavings = (monthly) => req(`/savings?monthly_requests=${monthly}`);
export const clearCache = () => req("/cache", { method: "DELETE" });
