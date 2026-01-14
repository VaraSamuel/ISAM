import React, { useMemo, useState } from "react";
import ThresholdSlider from "./components/ThresholdSlider.jsx";
import Results from "./components/Results.jsx";
import MetricHelp from "./components/MetricHelp.jsx";
import { createRun, subclassifyMany } from "./api";

const LAB_URL = import.meta.env.VITE_LAB_URL || "";

function describeRun({
  tStart,
  tEnd,
  dt,
  k,
  activityMethod,
  zThresh,
  minAboveSec,
}) {
  const w = `${Number(tStart).toFixed(1)}–${Number(tEnd).toFixed(1)}s (dt=${dt})`;
  const m = (activityMethod || "z_duration").toLowerCase();

  const lines = [];
  lines.push(`We slice each neuron trace to ${w}.`);

  if (m === "z_duration") {
    lines.push(
      `A neuron is active only if z(t) ≥ ${Number(zThresh).toFixed(
        2
      )} for at least ${Number(minAboveSec).toFixed(1)} seconds (continuous).`
    );
    lines.push(`This active/non-active filter is applied before clustering.`);
  } else {
    lines.push(`We label active neurons using method: ${activityMethod}.`);
  }

  lines.push(`Then we cluster only the active traces into k=${k} groups.`);
  return lines;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // time window + dt
  const [tStart, setTStart] = useState(0.0);
  const [tEnd, setTEnd] = useState(10.0);
  const [dt, setDt] = useState(1.0);

  // activity detection
  const [activityMethod, setActivityMethod] = useState("z_duration");
  const [zThresh, setZThresh] = useState(2.5);
  const [minAboveSec, setMinAboveSec] = useState(10.0);

  // keep existing knobs (optional)
  const [aucThresh, setAucThresh] = useState(0.0);
  const [meanThresh, setMeanThresh] = useState(0.0);
  const [promThresh, setPromThresh] = useState(0.0);
  const [baselineFrac, setBaselineFrac] = useState(0.1);
  const [minPeaks, setMinPeaks] = useState(1);

  // clustering
  const [k, setK] = useState(3);

  // selections for subclassify
  const [selected, setSelected] = useState({});

  const maxSeconds = 60;
  const presets = [
    [0, 1],
    [1, 10],
    [10, 20],
    [20, 30],
  ];

  const narrativeLines = useMemo(
    () =>
      describeRun({
        tStart,
        tEnd,
        dt,
        k,
        activityMethod,
        zThresh,
        minAboveSec,
      }),
    [tStart, tEnd, dt, k, activityMethod, zThresh, minAboveSec]
  );

  async function runInitial() {
    if (!file) return;
    setError("");
    setLoading(true);
    try {
      const r = await createRun(file, k, {
        t_start: tStart,
        t_end: tEnd,
        dt,

        activity_method: activityMethod,
        z_thresh: zThresh,
        min_above_sec: minAboveSec,

        auc_thresh: aucThresh,
        mean_thresh: meanThresh,
        prom_thresh: promThresh,
        baseline_frac: baselineFrac,
        min_peaks: minPeaks,
      });
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
      const r = await subclassifyMany(result.run_id, keys, k, {
        t_start: tStart,
        t_end: tEnd,
        dt,

        activity_method: activityMethod,
        z_thresh: zThresh,
        min_above_sec: minAboveSec,

        auc_thresh: aucThresh,
        mean_thresh: meanThresh,
        prom_thresh: promThresh,
        baseline_frac: baselineFrac,
        min_peaks: minPeaks,
      });
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
      <div className="topbar" style={{ flexDirection: "column", gap: 6 }}>
        <div className="brand">iSAM</div>
        <div className="subtitle">INTERACTIVE SPIKE ACTIVITY MAPPER</div>
      </div>

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

        {/* K */}
        <div style={{ marginTop: 14 }}>
          <div className="sliderTop">
            <div className="strong">Clusters (k)</div>
            <div className="pill">{k}</div>
          </div>
          <input
            type="range"
            min={1}
            max={10}
            step={1}
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            disabled={loading}
          />
        </div>

        {/* Time Window */}
        <div style={{ marginTop: 18 }}>
          <div className="sliderTop">
            <div className="strong">Time Window</div>
            <div className="pill">
              {tStart.toFixed(1)}s – {tEnd.toFixed(1)}s
            </div>
          </div>

          <div className="muted" style={{ fontSize: 13, marginBottom: 6 }}>
            Start (seconds)
          </div>
          <input
            type="range"
            min={0}
            max={maxSeconds}
            step={0.1}
            value={tStart}
            onChange={(e) => {
              const v = Number(e.target.value);
              setTStart(Math.min(v, tEnd));
            }}
            disabled={loading}
          />

          <div className="muted" style={{ fontSize: 13, marginTop: 10, marginBottom: 6 }}>
            End (seconds)
          </div>
          <input
            type="range"
            min={0}
            max={maxSeconds}
            step={0.1}
            value={tEnd}
            onChange={(e) => {
              const v = Number(e.target.value);
              setTEnd(Math.max(v, tStart));
            }}
            disabled={loading}
          />

          <div className="actions" style={{ marginTop: 12 }}>
            {presets.map(([a, b]) => (
              <button
                key={`${a}-${b}`}
                className="btn-secondary"
                type="button"
                onClick={() => {
                  setTStart(a);
                  setTEnd(b);
                }}
                disabled={loading}
              >
                {a}–{b}s
              </button>
            ))}
          </div>

          <div style={{ marginTop: 12 }}>
            <div className="muted" style={{ fontSize: 13, marginBottom: 6 }}>
              dt (seconds per column)
            </div>
            <input
              type="number"
              step="0.001"
              min="0.0001"
              value={dt}
              onChange={(e) => setDt(Number(e.target.value) || 1.0)}
              className="textInput"
              disabled={loading}
            />
            <div className="muted" style={{ marginTop: 6, fontSize: 12 }}>
              If each column is 1 second, keep dt = 1. If sampled at 10Hz, dt = 0.1.
            </div>
          </div>
        </div>

        {/* Activity Detection + Help Sidebar */}
        <div style={{ marginTop: 18 }}>
          <div className="section-title">Activity Detection</div>

          <div className="activityGrid">
            {/* Controls */}
            <div className="activityControls">
              <div className="row2">
                <div>
                  <div className="muted" style={{ fontSize: 13, marginBottom: 6 }}>
                    Method
                  </div>
                  <select
                    value={activityMethod}
                    onChange={(e) => setActivityMethod(e.target.value)}
                    className="textInput"
                    disabled={loading}
                  >
                    <option value="z_duration">Threshold + Duration (z_duration)</option>
                    <option value="max_z">Window Z-score max (max_z)</option>
                    <option value="auc">Area under curve (auc)</option>
                    <option value="mean_over_baseline">Mean over baseline</option>
                    <option value="prominence">Peak prominence</option>
                    <option value="composite">Composite (robust OR)</option>
                  </select>
                </div>

                <div>
                  <div className="muted" style={{ fontSize: 13, marginBottom: 6 }}>
                    Baseline fraction
                  </div>
                  <input
                    type="number"
                    step="0.01"
                    min="0.01"
                    max="0.9"
                    value={baselineFrac}
                    onChange={(e) => setBaselineFrac(Number(e.target.value) || 0.1)}
                    className="textInput"
                    disabled={loading}
                  />
                </div>
              </div>

              {/* z threshold + min duration */}
              <div style={{ marginTop: 12 }} className="row2">
                <div>
                  <ThresholdSlider
                    label="Z threshold (z_thresh)"
                    value={zThresh}
                    setValue={setZThresh}
                    min={0}
                    max={10}
                    step={0.1}
                    disabled={loading}
                  />
                </div>

                <div>
                  <div className="muted" style={{ fontSize: 13, marginBottom: 6 }}>
                    Min duration above z (sec)
                  </div>
                  <input
                    type="number"
                    step="0.1"
                    min="0"
                    value={minAboveSec}
                    onChange={(e) => setMinAboveSec(Number(e.target.value) || 0)}
                    className="textInput"
                    disabled={loading}
                  />
                </div>
              </div>

              {/* optional thresholds */}
              <div style={{ marginTop: 12 }} className="row2">
                <div>
                  <div className="muted" style={{ fontSize: 13, marginBottom: 6 }}>
                    AUC threshold
                  </div>
                  <input
                    type="number"
                    step="0.01"
                    value={aucThresh}
                    onChange={(e) => setAucThresh(Number(e.target.value) || 0)}
                    className="textInput"
                    disabled={loading}
                  />
                </div>
                <div>
                  <div className="muted" style={{ fontSize: 13, marginBottom: 6 }}>
                    Mean-over-baseline threshold
                  </div>
                  <input
                    type="number"
                    step="0.01"
                    value={meanThresh}
                    onChange={(e) => setMeanThresh(Number(e.target.value) || 0)}
                    className="textInput"
                    disabled={loading}
                  />
                </div>
              </div>

              <div style={{ marginTop: 12 }} className="row2">
                <div>
                  <div className="muted" style={{ fontSize: 13, marginBottom: 6 }}>
                    Prominence threshold
                  </div>
                  <input
                    type="number"
                    step="0.01"
                    value={promThresh}
                    onChange={(e) => setPromThresh(Number(e.target.value) || 0)}
                    className="textInput"
                    disabled={loading}
                  />
                </div>

                <div>
                  <div className="muted" style={{ fontSize: 13, marginBottom: 6 }}>
                    Min peaks (for prominence)
                  </div>
                  <input
                    type="number"
                    step="1"
                    min="1"
                    value={minPeaks}
                    onChange={(e) => setMinPeaks(Math.max(1, Number(e.target.value) || 1))}
                    className="textInput"
                    disabled={loading}
                  />
                </div>
              </div>
            </div>

            {/* Help panel */}
            <MetricHelp
              activityMethod={activityMethod}
              zThresh={zThresh}
              minAboveSec={minAboveSec}
              aucThresh={aucThresh}
              meanThresh={meanThresh}
              promThresh={promThresh}
              baselineFrac={baselineFrac}
              minPeaks={minPeaks}
            />
          </div>
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

        <div className="narrativeBox">
          <div className="narrativeTitle">What will happen when you run?</div>
          <ul className="narrativeList">
            {narrativeLines.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>

        {error && <pre className="errorBox">{error}</pre>}
      </div>

      {result && (
        <Results
          result={result}
          loading={loading}
          selected={selected}
          setSelected={setSelected}
          onSubclassifySelected={subclassifySelected}
        />
      )}

      {/* Always show footer */}
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
    </div>
  );
}
