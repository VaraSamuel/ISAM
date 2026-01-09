import React, { useState } from "react";
import ThresholdSlider from "./components/ThresholdSlider.jsx";
import Results from "./components/Results.jsx";
import { createRun, subclassifyMany } from "./api";

const LAB_URL = import.meta.env.VITE_LAB_URL || ""; // set this in .env.local if you want a clickable link

export default function App() {
  const [file, setFile] = useState(null);
  const [threshold, setThreshold] = useState(2.5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // checkbox selections for multi-subclassify
  const [selected, setSelected] = useState({});

  async function runInitial() {
    if (!file) return;
    setError("");
    setLoading(true);
    try {
      const r = await createRun(file, threshold, 3);
      setResult(r);
      setSelected({});
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }

  async function subclassifySelected() {
    if (!result?.run_id) return;
    const keys = Object.keys(selected).filter((k) => selected[k]);
    if (keys.length === 0) return;

    setError("");
    setLoading(true);
    try {
      const r = await subclassifyMany(result.run_id, keys, 3);
      setResult(r);
      setSelected({});
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }

  const year = new Date().getFullYear();

  return (
    <div className="container">
      {/* Centered title + abbreviation */}
      <div className="topbar" style={{ flexDirection: "column", gap: 6 }}>
        <div className="brand">iSAM</div>
        <div className="subtitle">Interactive Spike Activity Mapper</div>
      </div>

      {/* Upload + Threshold */}
      <div className="card">
        <div className="section-title">Upload Excel</div>

        <input
          type="file"
          accept=".xlsx,.xls"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        {file && (
          <div className="muted" style={{ marginTop: 8, fontSize: 13 }}>
            {file.name}
          </div>
        )}

        <div style={{ marginTop: 16 }}>
          <ThresholdSlider value={threshold} setValue={setThreshold} min={0} max={10} step={0.1} />
        </div>

        <div className="actions" style={{ marginTop: 14 }}>
          <button onClick={runInitial} disabled={!file || loading}>
            {loading ? "Running..." : "Run clustering"}
          </button>

          <button
            className="btn-secondary"
            onClick={() => {
              setResult(null);
              setSelected({});
              setError("");
            }}
            disabled={loading}
          >
            Clear
          </button>
        </div>

        {error && <pre className="errorBox">{error}</pre>}
      </div>

      {/* Results */}
      {result && (
        <Results
          result={result}
          loading={loading}
          selected={selected}
          setSelected={setSelected}
          onSubclassifySelected={subclassifySelected}
        />
      )}

      {/* Footer ONLY after results exist */}
      {result && (
        <footer className="footer">
          <div className="footer-line">
            Developed by <span className="footer-strong">Samuel Vara</span>,{" "}
            {LAB_URL ? (
              <a className="footer-link" href={LAB_URL} target="_blank" rel="noreferrer">
                The De Lartigue Lab
              </a>
            ) : (
              <span className="footer-strong">The De Lartigue Lab</span>
            )}
          </div>
          <div className="footer-sub">© {year}</div>
        </footer>
      )}
    </div>
  );
}
