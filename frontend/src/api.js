const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function createRun(file, threshold, k = 3) {
  const form = new FormData();
  form.append("excel_file", file);
  form.append("threshold", String(threshold));
  form.append("k", String(k));

  const res = await fetch(`${API_BASE}/runs`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function subclassifyMany(runId, selectedClusters, k = 3) {
  const res = await fetch(`${API_BASE}/runs/subclassify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: runId, selected_clusters: selectedClusters, k }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function artifactUrl(relativePath) {
  return `${API_BASE}${relativePath}`;
}
