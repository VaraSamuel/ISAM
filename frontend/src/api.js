const API_BASE = import.meta.env.VITE_API_BASE || "";

export function artifactUrl(path) {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE}${path}`;
}

async function asJson(res) {
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {}
  if (!res.ok) {
    const msg = data?.detail || data?.message || text || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

export async function createRun(file, k, opts = {}) {
  const fd = new FormData();
  fd.append("excel_file", file);
  fd.append("k", String(k));

  fd.append("t_start", String(opts.t_start ?? 0.0));
  fd.append("t_end", String(opts.t_end ?? -1.0));
  fd.append("dt", String(opts.dt ?? 1.0));

  fd.append("activity_method", String(opts.activity_method ?? "z_duration"));
  fd.append("z_thresh", String(opts.z_thresh ?? 2.5));
  fd.append("min_above_sec", String(opts.min_above_sec ?? 10.0));

  fd.append("auc_thresh", String(opts.auc_thresh ?? 0.0));
  fd.append("mean_thresh", String(opts.mean_thresh ?? 0.0));
  fd.append("prom_thresh", String(opts.prom_thresh ?? 0.0));
  fd.append("baseline_frac", String(opts.baseline_frac ?? 0.1));
  fd.append("min_peaks", String(opts.min_peaks ?? 1));

  const res = await fetch(`${API_BASE}/runs`, { method: "POST", body: fd });
  return asJson(res);
}

export async function subclassifyMany(run_id, selected_clusters, k, opts = {}) {
  const payload = {
    run_id,
    selected_clusters,
    k,

    t_start: opts.t_start ?? 0.0,
    t_end: opts.t_end ?? -1.0,
    dt: opts.dt ?? 1.0,

    activity_method: opts.activity_method ?? "z_duration",
    z_thresh: opts.z_thresh ?? 2.5,
    min_above_sec: opts.min_above_sec ?? 10.0,

    auc_thresh: opts.auc_thresh ?? 0.0,
    mean_thresh: opts.mean_thresh ?? 0.0,
    prom_thresh: opts.prom_thresh ?? 0.0,
    baseline_frac: opts.baseline_frac ?? 0.1,
    min_peaks: opts.min_peaks ?? 1,
  };

  const res = await fetch(`${API_BASE}/runs/subclassify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  return asJson(res);
}
