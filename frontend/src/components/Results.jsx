import React, { useMemo } from "react";
import { artifactUrl } from "../api";

function explainResult(result) {
  const { counts, k, t_start, t_end, dt, activity_method, thresholds } = result;
  const m = (activity_method || "z_duration").toLowerCase();
  const w = `${Number(t_start).toFixed(1)}–${Number(t_end).toFixed(1)}s (dt=${dt})`;

  const s = [];
  s.push(`We analyzed ${counts.total} neurons in the time window ${w}.`);
  s.push(`We labeled ${counts.active} as active and ${counts.non_active} as non-active before clustering.`);

  if (m === "z_duration") {
    s.push(
      `Rule: z(t) ≥ ${Number(thresholds?.z_thresh ?? 0).toFixed(2)} continuously for ≥ ${Number(
        thresholds?.min_above_sec ?? 0
      ).toFixed(1)}s.`
    );
  } else {
    s.push(`Method: ${activity_method}.`);
  }

  s.push(`Then we clustered active traces into k=${k} groups and generated plots + Excel exports.`);
  return s;
}

export default function Results({ result, loading, selected, setSelected, onSubclassifySelected }) {
  const { run_id, counts, clusters, artifacts, activity_method, thresholds, t_start, t_end, dt } = result;
  const explanation = useMemo(() => explainResult(result), [result]);

  return (
    <div className="card" style={{ marginTop: 18 }}>
      <div className="summaryRow">
        <div className="mono">{run_id}</div>
        <div className="muted">
          Active {counts.active} | Non-Active {counts.non_active} | Total {counts.total}
        </div>
      </div>

      <div className="muted" style={{ marginTop: 8, fontSize: 13 }}>
        Time window: {t_start}s – {t_end}s | dt={dt} | method={activity_method}
      </div>

      <div className="muted" style={{ marginTop: 6, fontSize: 12 }}>
        thresholds: z={thresholds?.z_thresh} | min_above_sec={thresholds?.min_above_sec} | auc={thresholds?.auc_thresh} | mean={thresholds?.mean_thresh} | prom={thresholds?.prom_thresh} | baseline={thresholds?.baseline_frac} | min_peaks={thresholds?.min_peaks}
      </div>

      <div className="narrativeBox" style={{ marginTop: 12 }}>
        <div className="narrativeTitle">What happened?</div>
        <ul className="narrativeList">
          {explanation.map((x, i) => (
            <li key={i}>{x}</li>
          ))}
        </ul>
      </div>

      <hr />

      <div className="downloadRow">
        <a href={artifactUrl(artifacts.active_excel)} target="_blank" rel="noreferrer">
          Download Active.xlsx
        </a>
        <a href={artifactUrl(artifacts.non_active_excel)} target="_blank" rel="noreferrer">
          Download Non-Active.xlsx
        </a>
        <a href={artifactUrl(artifacts.metrics_csv)} target="_blank" rel="noreferrer">
          Download metrics.csv
        </a>
      </div>

      <hr />

      <div className="section-title">Clusters</div>

      <div className="actions" style={{ marginBottom: 10 }}>
        <button onClick={onSubclassifySelected} disabled={loading}>
          Subclassify Selected Clusters
        </button>
      </div>

      <div className="grid">
        {clusters.map((c) => {
          const key = c.id;
          const plots = artifacts.cluster_plots?.[key];
          const excel = artifacts.cluster_excels?.[key];

          return (
            <div key={key} className="clusterCard">
              <div className="clusterHeader">
                <div className="clusterId">{key}</div>
                <div className="clusterSize muted">{c.size} neurons</div>
              </div>

              <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 10 }}>
                <input
                  type="checkbox"
                  checked={!!selected[key]}
                  onChange={(e) => setSelected((prev) => ({ ...prev, [key]: e.target.checked }))}
                />
                <div className="muted">Select for subclassify</div>
              </div>

              <div style={{ marginTop: 10 }}>
                <a href={artifactUrl(excel)} target="_blank" rel="noreferrer">
                  Download Excel
                </a>
              </div>

              <div style={{ marginTop: 12 }}>
                <div className="plotLabel">All Neuron Traces</div>
                {plots?.all_traces ? (
                  <img className="plot-img" src={artifactUrl(plots.all_traces)} alt={`all_traces_${key}`} />
                ) : (
                  <div className="muted">No plot</div>
                )}
              </div>

              <div style={{ marginTop: 12 }}>
                <div className="plotLabel">Average Trace</div>
                {plots?.avg_trace ? (
                  <img className="plot-img" src={artifactUrl(plots.avg_trace)} alt={`avg_trace_${key}`} />
                ) : (
                  <div className="muted">No plot</div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
