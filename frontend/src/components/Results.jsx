import React from "react";
import { artifactUrl } from "../api";

function prettyClusterId(id) {
  // "1" -> "1"
  // "10" -> "1-0"
  // "121" -> "12-1"
  if (!id) return id;
  if (id.length <= 1) return id;
  return `${id.slice(0, -1)}-${id.slice(-1)}`;
}

export default function Results({
  result,
  loading,
  selected,
  setSelected,
  onSubclassifySelected,
}) {
  const { run_id, threshold, counts, clusters, artifacts } = result;
  const anySelected = Object.values(selected || {}).some(Boolean);

  return (
    <div style={{ marginTop: 16 }} className="card">
      {/* Run summary row */}
      <div className="summaryRow">
        <div>
          <div className="muted" style={{ fontSize: 12 }}>Run</div>
          <div className="mono strong">{run_id}</div>
        </div>

        <div>
          <div className="muted" style={{ fontSize: 12 }}>Threshold</div>
          <div className="strong">{threshold}</div>
        </div>

        <div>
          <div className="muted" style={{ fontSize: 12 }}>Active / Non-active</div>
          <div className="strong">
            {counts?.active ?? 0} / {counts?.non_active ?? 0} (total {counts?.total ?? 0})
          </div>
        </div>
      </div>

      <hr />

      {/* Downloads */}
      <div className="downloadRow">
        {artifacts?.active_excel && (
          <a href={artifactUrl(artifacts.active_excel)} target="_blank" rel="noreferrer">
            Download Active.xlsx
          </a>
        )}
        {artifacts?.non_active_excel && (
          <a href={artifactUrl(artifacts.non_active_excel)} target="_blank" rel="noreferrer">
            Download Non-Active.xlsx
          </a>
        )}
      </div>

      <div style={{ marginTop: 14 }} className="section-title">Clusters</div>

      <div className="actions" style={{ marginTop: 10, marginBottom: 12 }}>
        <button
          onClick={onSubclassifySelected}
          disabled={loading || !anySelected}
        >
          {loading ? "Working..." : "Subclassify Selected Clusters"}
        </button>
      </div>

      {(!clusters || clusters.length === 0) && (
        <div className="muted">No clusters found.</div>
      )}

      {/* Cluster cards */}
      <div className="grid">
        {clusters?.map((c) => {
          const cp = artifacts?.cluster_plots?.[c.id];

          return (
            <div key={c.id} className="clusterCard">
              <div className="clusterHeader">
                <input
                  type="checkbox"
                  checked={!!selected?.[c.id]}
                  onChange={(e) =>
                    setSelected((prev) => ({ ...(prev || {}), [c.id]: e.target.checked }))
                  }
                />

                <div className="clusterId">{prettyClusterId(c.id)}</div>

                <div className="clusterSize muted">{c.size} neurons</div>
              </div>

              <div style={{ marginTop: 8 }}>
                {artifacts?.cluster_excels?.[c.id] && (
                  <a
                    href={artifactUrl(artifacts.cluster_excels[c.id])}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Download Excel
                  </a>
                )}
              </div>

              {/* EXACT plots you wanted */}
              {cp?.all_traces && (
                <div style={{ marginTop: 12 }}>
                  <div className="plotLabel">All Neuron Traces:</div>
                  <img
                    className="plot-img"
                    src={artifactUrl(cp.all_traces)}
                    alt={`All Traces ${prettyClusterId(c.id)}`}
                  />
                </div>
              )}

              {cp?.avg_trace && (
                <div style={{ marginTop: 14 }}>
                  <div className="plotLabel">Average Trace:</div>
                  <img
                    className="plot-img"
                    src={artifactUrl(cp.avg_trace)}
                    alt={`Average Trace ${prettyClusterId(c.id)}`}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
