# backend/app/pipeline.py
import os
import json
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from scipy.signal import find_peaks, peak_prominences


# -----------------------------
# URLs for artifacts served by FastAPI StaticFiles
# -----------------------------
def _artifact_url(run_id: str, rel_path: str) -> str:
    return f"/artifacts/{run_id}/{rel_path}".replace("\\", "/")


def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -----------------------------
# Time window helper
# -----------------------------
def _apply_time_window(
    X: np.ndarray,
    t: np.ndarray,
    t_start: float,
    t_end: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if t.size == 0:
        return X, t

    t_start = float(max(0.0, t_start))
    if t_end is None or float(t_end) < 0:
        t_end = float(t[-1])

    t_end = float(max(t_start, float(t_end)))

    mask = (t >= t_start) & (t <= t_end)
    if not np.any(mask):
        # no overlap → return full arrays (better than empty)
        return X, t

    return X[:, mask], t[mask]


# -----------------------------
# Excel -> traces
# -----------------------------
def extract_traces_from_excel(df: pd.DataFrame, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    X = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    T = X.shape[1]
    t = np.arange(T, dtype=float) * float(dt)
    return X, t


# -----------------------------
# Activity metrics
# -----------------------------
def _compute_window_metrics(
    Xw: np.ndarray,
    t: np.ndarray,
    baseline_frac: float = 0.1,
    min_peaks: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Compute per-cell metrics inside the window.

    Returns dict of vectors length N:
      - max_z: max window-relative z
      - auc: integral of baseline-subtracted signal over time
      - mean_over_baseline: mean of baseline-subtracted signal
      - prominence: max peak prominence of baseline-subtracted signal
      - n_peaks: number of peaks found
    """
    N, T = Xw.shape
    if T == 0:
        zeros = np.zeros(N, dtype=float)
        return {
            "max_z": zeros,
            "auc": zeros,
            "mean_over_baseline": zeros,
            "prominence": zeros,
            "n_peaks": zeros,
        }

    baseline_frac = float(np.clip(baseline_frac, 0.01, 0.9))
    bT = max(1, int(np.floor(T * baseline_frac)))
    baseline = Xw[:, :bT].mean(axis=1, keepdims=True)

    Xrel = Xw - baseline

    mu = Xw.mean(axis=1, keepdims=True)
    sigma = Xw.std(axis=1, keepdims=True) + 1e-6
    Xz = (Xw - mu) / sigma
    max_z = Xz.max(axis=1)

    auc = np.trapz(Xrel, t, axis=1)
    mean_over_baseline = Xrel.mean(axis=1)

    prominence = np.zeros(N, dtype=float)
    n_peaks_vec = np.zeros(N, dtype=float)

    for i in range(N):
        y = Xrel[i]
        peaks, _ = find_peaks(y)
        n_peaks_vec[i] = float(len(peaks))
        if len(peaks) >= int(min_peaks):
            proms = peak_prominences(y, peaks)[0]
            prominence[i] = float(proms.max()) if len(proms) else 0.0
        else:
            prominence[i] = 0.0

    return {
        "max_z": max_z.astype(float),
        "auc": auc.astype(float),
        "mean_over_baseline": mean_over_baseline.astype(float),
        "prominence": prominence.astype(float),
        "n_peaks": n_peaks_vec.astype(float),
    }


def _max_run_sec_above_z(Xw: np.ndarray, z_thresh: float, dt: float) -> np.ndarray:
    """
    Per neuron, compute max contiguous duration (sec) where window-relative z >= z_thresh.
    """
    N, T = Xw.shape
    if T == 0:
        return np.zeros(N, dtype=float)

    mu = Xw.mean(axis=1, keepdims=True)
    sigma = Xw.std(axis=1, keepdims=True) + 1e-6
    Xz = (Xw - mu) / sigma
    above = Xz >= float(z_thresh)

    max_run = np.zeros(N, dtype=float)
    step = float(dt)

    for i in range(N):
        row = above[i]
        best = 0
        cur = 0
        for v in row:
            if v:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        max_run[i] = best * step

    return max_run


def _active_mask_from_metrics(
    *,
    metrics: Dict[str, np.ndarray],
    activity_method: str,
    z_thresh: float,
    auc_thresh: float,
    mean_thresh: float,
    prom_thresh: float,
    min_peaks: int,
    Xw: Optional[np.ndarray] = None,
    dt: float = 1.0,
    min_above_sec: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (active_mask, max_run_sec).
    max_run_sec is computed only when needed (or for composite if enabled).
    """
    method = (activity_method or "max_z").strip().lower()

    max_z = metrics["max_z"]
    auc = metrics["auc"]
    mean_over_baseline = metrics["mean_over_baseline"]
    prom = metrics["prominence"]
    n_peaks = metrics["n_peaks"]

    max_run_sec = np.zeros_like(max_z, dtype=float)

    if method == "z_duration":
        if Xw is None:
            return (max_z >= float(z_thresh)), max_run_sec
        max_run_sec = _max_run_sec_above_z(Xw, z_thresh=float(z_thresh), dt=float(dt))
        active = (max_z >= float(z_thresh)) & (max_run_sec >= float(min_above_sec))
        return active, max_run_sec

    if method == "max_z":
        return (max_z >= float(z_thresh)), max_run_sec

    if method == "auc":
        return (auc >= float(auc_thresh)), max_run_sec

    if method == "mean_over_baseline":
        return (mean_over_baseline >= float(mean_thresh)), max_run_sec

    if method == "prominence":
        return ((prom >= float(prom_thresh)) & (n_peaks >= float(min_peaks))), max_run_sec

    if method == "composite":
        dur_ok = np.zeros_like(max_z, dtype=bool)
        if Xw is not None and float(min_above_sec) > 0:
            max_run_sec = _max_run_sec_above_z(Xw, z_thresh=float(z_thresh), dt=float(dt))
            dur_ok = (max_z >= float(z_thresh)) & (max_run_sec >= float(min_above_sec))

        active = (
            (max_z >= float(z_thresh)) |
            (auc >= float(auc_thresh)) |
            (mean_over_baseline >= float(mean_thresh)) |
            ((prom >= float(prom_thresh)) & (n_peaks >= float(min_peaks))) |
            dur_ok
        )
        return active, max_run_sec

    return (max_z >= float(z_thresh)), max_run_sec


# -----------------------------
# Plot helpers
# -----------------------------
def _robust_ylim(Xc: np.ndarray) -> Tuple[float, float]:
    hi = float(np.percentile(Xc, 99.5)) if Xc.size else 1.0
    if hi <= 0:
        hi = float(np.max(Xc)) if Xc.size else 1.0
    return 0.0, max(hi * 1.08, 1.0)


def plot_all_traces_png(Xc: np.ndarray, t: np.ndarray, out_path: str, title: str) -> None:
    y0, y1 = _robust_ylim(Xc)

    fig = plt.figure(figsize=(10, 5.5))
    ax = plt.gca()
    ax.set_facecolor("white")

    for y in Xc:
        ax.plot(t, y, linewidth=1.3, alpha=0.25)

    ax.set_title(title, fontsize=20, fontweight="bold", pad=10)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Z-score", fontsize=14)
    ax.set_ylim(y0, y1)
    ax.grid(True, alpha=0.6)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_avg_trace_png(Xc: np.ndarray, t: np.ndarray, out_path: str, title: str) -> None:
    avg = Xc.mean(axis=0) if Xc.size else np.zeros_like(t)
    y0, y1 = _robust_ylim(Xc)

    fig = plt.figure(figsize=(10, 5.5))
    ax = plt.gca()
    ax.set_facecolor("white")

    ax.plot(t, avg, linewidth=3.2)

    ax.set_title(title, fontsize=20, fontweight="bold", pad=10)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Z-score", fontsize=14)
    ax.set_ylim(y0, y1)
    ax.grid(True, alpha=0.6)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Clustering utilities
# -----------------------------
def _kmeans_cluster(Xc: np.ndarray, k: int = 3, seed: int = 0) -> np.ndarray:
    n = Xc.shape[0]
    k_eff = max(1, min(k, n))
    if k_eff == 1:
        return np.zeros(n, dtype=int)

    km = KMeans(n_clusters=k_eff, random_state=seed, n_init="auto")
    labels = km.fit_predict(Xc)
    return labels.astype(int)


def _make_child_keys(prefix: str, n_children: int) -> List[str]:
    if prefix == "":
        return [str(i + 1) for i in range(n_children)]
    return [f"{prefix}{i}" for i in range(n_children)]


def _cluster_root(
    *,
    X: np.ndarray,
    active_idx: np.ndarray,
    k: int,
) -> Dict[str, List[int]]:
    """
    Root clustering only. Safe if active_idx is empty.
    """
    if active_idx.size == 0:
        return {}

    Xc = X[active_idx]
    if Xc.shape[0] == 0:
        return {}

    labels = _kmeans_cluster(Xc, k=k, seed=0)
    n_children = int(labels.max()) + 1 if labels.size else 0
    if n_children <= 0:
        return {}

    keys = _make_child_keys("", n_children)

    clusters: Dict[str, List[int]] = {}
    for i, key in enumerate(keys):
        members = active_idx[labels == i].tolist()
        clusters[key] = members
    return clusters


def _cluster_subdivide(
    *,
    X: np.ndarray,
    clusters: Dict[str, List[int]],
    key: str,
    k: int,
) -> Dict[str, List[int]]:
    """
    Replace one cluster key with its children clusters. Safe for small/empty.
    """
    idx_list = clusters.get(key, [])
    if len(idx_list) < 3:
        return clusters

    target_idx = np.array(idx_list, dtype=int)
    Xc = X[target_idx]
    if Xc.shape[0] == 0:
        return clusters

    labels = _kmeans_cluster(Xc, k=k, seed=0)
    n_children = int(labels.max()) + 1 if labels.size else 0
    if n_children <= 0:
        return clusters

    children_keys = _make_child_keys(key, n_children)

    # remove parent
    clusters = dict(clusters)
    clusters.pop(key, None)

    for i, ck in enumerate(children_keys):
        clusters[ck] = target_idx[labels == i].tolist()

    return clusters


# -----------------------------
# Export cluster excels + plots + metrics
# -----------------------------
def _export_cluster_excels(
    df_raw: pd.DataFrame,
    clusters: Dict[str, List[int]],
    clusters_dir: str,
    run_id: str
) -> Dict[str, str]:
    os.makedirs(clusters_dir, exist_ok=True)
    mapping: Dict[str, str] = {}

    for key, idx_list in clusters.items():
        cdf = df_raw.iloc[idx_list].copy()
        path = os.path.join(clusters_dir, f"cluster_{key}.xlsx")
        cdf.to_excel(path, index=False)
        mapping[key] = _artifact_url(run_id, f"clusters/cluster_{key}.xlsx")

    return mapping


def _write_cluster_plots(
    X: np.ndarray,
    t: np.ndarray,
    clusters: Dict[str, List[int]],
    plots_dir: str,
    run_id: str,
) -> Dict[str, Dict[str, str]]:
    os.makedirs(plots_dir, exist_ok=True)
    out: Dict[str, Dict[str, str]] = {}

    for key, idx_list in clusters.items():
        idx = np.array(idx_list, dtype=int)
        if idx.size == 0:
            continue

        Xc = X[idx]
        if Xc.shape[0] == 0:
            continue

        all_path = os.path.join(plots_dir, f"all_traces_{key}.png")
        avg_path = os.path.join(plots_dir, f"avg_trace_{key}.png")

        plot_all_traces_png(Xc, t, all_path, title=f"All Traces - {key}")
        plot_avg_trace_png(Xc, t, avg_path, title=f"Average Trace - {key}")

        out[key] = {
            "all_traces": _artifact_url(run_id, f"plots/all_traces_{key}.png"),
            "avg_trace": _artifact_url(run_id, f"plots/avg_trace_{key}.png"),
        }

    return out


def _write_metrics_csv(
    metrics_df: pd.DataFrame,
    out_dir: str,
    run_id: str
) -> str:
    path = os.path.join(out_dir, "metrics.csv")
    metrics_df.to_csv(path, index=False)
    return _artifact_url(run_id, "metrics.csv")


# -----------------------------
# API entrypoints
# -----------------------------
def create_new_run(
    *,
    excel_path: str,
    out_dir: str,
    run_id: str,
    k: int = 3,

    # time slicing
    t_start: float = 0.0,
    t_end: float = -1.0,
    dt: float = 1.0,

    # activity detection config
    activity_method: str = "max_z",
    z_thresh: float = 2.5,
    min_above_sec: float = 10.0,
    auc_thresh: float = 0.0,
    mean_thresh: float = 0.0,
    prom_thresh: float = 0.0,
    baseline_frac: float = 0.1,
    min_peaks: int = 1,
) -> Dict[str, Any]:
    plots_dir = os.path.join(out_dir, "plots")
    clusters_dir = os.path.join(out_dir, "clusters")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(clusters_dir, exist_ok=True)

    df_raw = pd.read_excel(excel_path)

    X_full, t_full = extract_traces_from_excel(df_raw, dt=dt)
    Xw, tw = _apply_time_window(X_full, t_full, t_start, t_end)

    metrics = _compute_window_metrics(Xw, tw, baseline_frac=baseline_frac, min_peaks=min_peaks)

    active_mask, max_run_sec = _active_mask_from_metrics(
        metrics=metrics,
        activity_method=activity_method,
        z_thresh=z_thresh,
        auc_thresh=auc_thresh,
        mean_thresh=mean_thresh,
        prom_thresh=prom_thresh,
        min_peaks=min_peaks,
        Xw=Xw,
        dt=dt,
        min_above_sec=min_above_sec,
    )

    active_idx = np.where(active_mask)[0]
    non_active_idx = np.where(~active_mask)[0]

    # export active/non-active
    df_raw.iloc[active_idx].to_excel(os.path.join(clusters_dir, "active.xlsx"), index=False)
    df_raw.iloc[non_active_idx].to_excel(os.path.join(clusters_dir, "non_active.xlsx"), index=False)

    # root clustering (SAFE if no active neurons)
    clusters = _cluster_root(X=Xw, active_idx=active_idx, k=k)

    cluster_excels = _export_cluster_excels(df_raw, clusters, clusters_dir, run_id) if clusters else {}
    cluster_plots = _write_cluster_plots(Xw, tw, clusters, plots_dir, run_id) if clusters else {}

    metrics_df = pd.DataFrame({
        "row_index": np.arange(len(df_raw)),
        "active": active_mask.astype(int),
        "max_z": metrics["max_z"],
        "max_run_sec": max_run_sec.astype(float),
        "auc": metrics["auc"],
        "mean_over_baseline": metrics["mean_over_baseline"],
        "prominence": metrics["prominence"],
        "n_peaks": metrics["n_peaks"],
    })
    metrics_csv = _write_metrics_csv(metrics_df, out_dir, run_id)

    _save_json(os.path.join(out_dir, "state.json"), {
        "k": int(k),
        "dt": float(dt),
        "t_start": float(t_start),
        "t_end": float(t_end),

        "activity_method": activity_method,
        "z_thresh": float(z_thresh),
        "min_above_sec": float(min_above_sec),
        "auc_thresh": float(auc_thresh),
        "mean_thresh": float(mean_thresh),
        "prom_thresh": float(prom_thresh),
        "baseline_frac": float(baseline_frac),
        "min_peaks": int(min_peaks),

        "active_idx": active_idx.tolist(),
        "clusters": clusters,
    })

    clusters_list = [{"id": key, "size": len(idx)} for key, idx in sorted(clusters.items())]

    summary = {
        "run_id": run_id,
        "k": int(k),
        "dt": float(dt),
        "t_start": float(t_start),
        "t_end": float(t_end),

        "activity_method": activity_method,
        "thresholds": {
            "z_thresh": float(z_thresh),
            "min_above_sec": float(min_above_sec),
            "auc_thresh": float(auc_thresh),
            "mean_thresh": float(mean_thresh),
            "prom_thresh": float(prom_thresh),
            "baseline_frac": float(baseline_frac),
            "min_peaks": int(min_peaks),
        },

        "counts": {
            "total": int(len(df_raw)),
            "active": int(len(active_idx)),
            "non_active": int(len(non_active_idx)),
        },
        "clusters": clusters_list,

        "artifacts": {
            "active_excel": _artifact_url(run_id, "clusters/active.xlsx"),
            "non_active_excel": _artifact_url(run_id, "clusters/non_active.xlsx"),
            "cluster_excels": cluster_excels,
            "cluster_plots": cluster_plots,
            "metrics_csv": metrics_csv,
        },
        "notes": [],
    }

    # helpful note for z_duration if window < min_above_sec
    win_len = float(tw[-1] - tw[0]) if tw.size >= 2 else float(tw[-1]) if tw.size == 1 else 0.0
    if (activity_method or "").strip().lower() == "z_duration" and float(min_above_sec) > win_len:
        summary["notes"].append(
            f"z_duration requires min_above_sec <= window_length. "
            f"window_length≈{win_len:.2f}s, min_above_sec={float(min_above_sec):.2f}s → likely 0 active neurons."
        )

    _save_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def subclassify_selected(
    *,
    out_dir: str,
    run_id: str,
    selected_clusters: List[str],
    k: int = 3,

    # time slicing
    t_start: float = 0.0,
    t_end: float = -1.0,
    dt: float = 1.0,

    # activity detection config
    activity_method: str = "max_z",
    z_thresh: float = 2.5,
    min_above_sec: float = 10.0,
    auc_thresh: float = 0.0,
    mean_thresh: float = 0.0,
    prom_thresh: float = 0.0,
    baseline_frac: float = 0.1,
    min_peaks: int = 1,
) -> Dict[str, Any]:
    state_path = os.path.join(out_dir, "state.json")
    state = _load_json(state_path, default=None)
    if not state:
        raise RuntimeError("Run state not found. Create a run first.")

    clusters: Dict[str, List[int]] = state.get("clusters", {})
    excel_path = os.path.join(out_dir, "input.xlsx")
    df_raw = pd.read_excel(excel_path)

    X_full, t_full = extract_traces_from_excel(df_raw, dt=dt)
    Xw, tw = _apply_time_window(X_full, t_full, t_start, t_end)

    metrics = _compute_window_metrics(Xw, tw, baseline_frac=baseline_frac, min_peaks=min_peaks)
    active_mask, max_run_sec = _active_mask_from_metrics(
        metrics=metrics,
        activity_method=activity_method,
        z_thresh=z_thresh,
        auc_thresh=auc_thresh,
        mean_thresh=mean_thresh,
        prom_thresh=prom_thresh,
        min_peaks=min_peaks,
        Xw=Xw,
        dt=dt,
        min_above_sec=min_above_sec,
    )
    active_idx = np.where(active_mask)[0]

    clusters_dir = os.path.join(out_dir, "clusters")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(clusters_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    df_raw.iloc[active_idx].to_excel(os.path.join(clusters_dir, "active.xlsx"), index=False)
    df_raw.iloc[np.where(~active_mask)[0]].to_excel(os.path.join(clusters_dir, "non_active.xlsx"), index=False)

    # If no clusters exist yet (e.g., previously 0 active), seed root clusters now
    if not clusters:
        clusters = _cluster_root(X=Xw, active_idx=active_idx, k=k)

    # Subdivide selected clusters
    for key in selected_clusters:
        clusters = _cluster_subdivide(X=Xw, clusters=clusters, key=key, k=k)

    cluster_excels = _export_cluster_excels(df_raw, clusters, clusters_dir, run_id) if clusters else {}
    cluster_plots = _write_cluster_plots(Xw, tw, clusters, plots_dir, run_id) if clusters else {}

    metrics_df = pd.DataFrame({
        "row_index": np.arange(len(df_raw)),
        "active": active_mask.astype(int),
        "max_z": metrics["max_z"],
        "max_run_sec": max_run_sec.astype(float),
        "auc": metrics["auc"],
        "mean_over_baseline": metrics["mean_over_baseline"],
        "prominence": metrics["prominence"],
        "n_peaks": metrics["n_peaks"],
    })
    metrics_csv = _write_metrics_csv(metrics_df, out_dir, run_id)

    _save_json(state_path, {
        "k": int(k),
        "dt": float(dt),
        "t_start": float(t_start),
        "t_end": float(t_end),

        "activity_method": activity_method,
        "z_thresh": float(z_thresh),
        "min_above_sec": float(min_above_sec),
        "auc_thresh": float(auc_thresh),
        "mean_thresh": float(mean_thresh),
        "prom_thresh": float(prom_thresh),
        "baseline_frac": float(baseline_frac),
        "min_peaks": int(min_peaks),

        "active_idx": active_idx.tolist(),
        "clusters": clusters,
    })

    clusters_list = [{"id": key, "size": len(idx)} for key, idx in sorted(clusters.items())]

    summary = {
        "run_id": run_id,
        "k": int(k),
        "dt": float(dt),
        "t_start": float(t_start),
        "t_end": float(t_end),

        "activity_method": activity_method,
        "thresholds": {
            "z_thresh": float(z_thresh),
            "min_above_sec": float(min_above_sec),
            "auc_thresh": float(auc_thresh),
            "mean_thresh": float(mean_thresh),
            "prom_thresh": float(prom_thresh),
            "baseline_frac": float(baseline_frac),
            "min_peaks": int(min_peaks),
        },

        "counts": {
            "total": int(len(df_raw)),
            "active": int(len(active_idx)),
            "non_active": int(len(df_raw) - len(active_idx)),
        },
        "clusters": clusters_list,

        "artifacts": {
            "active_excel": _artifact_url(run_id, "clusters/active.xlsx"),
            "non_active_excel": _artifact_url(run_id, "clusters/non_active.xlsx"),
            "cluster_excels": cluster_excels,
            "cluster_plots": cluster_plots,
            "metrics_csv": metrics_csv,
        },
    }

    _save_json(os.path.join(out_dir, "summary.json"), summary)
    return summary
