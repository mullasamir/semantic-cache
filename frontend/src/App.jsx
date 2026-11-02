import { useState } from "react";
import Playground from "./Playground.jsx";
import Dashboard from "./Dashboard.jsx";

export default function App() {
  const [tab, setTab] = useState("playground");
  return (
    <div className="shell">
      <header className="top">
        <h1>Semantic cache</h1>
        <nav role="tablist" aria-label="Views">
          {[
            ["playground", "Playground"],
            ["dashboard", "Dashboard"],
          ].map(([id, label]) => (
            <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? "tab on" : "tab"} onClick={() => setTab(id)}>
              {label}
            </button>
          ))}
        </nav>
      </header>
      <main>{tab === "playground" ? <Playground /> : <Dashboard />}</main>
    </div>
  );
}
