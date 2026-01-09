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
# Excel -> traces
# IMPORTANT: to match your reference plots, we force time axis = 0..T-1
# (NOT parsing column names, which caused 0..3 scaling for some files)
# -----------------------------
def extract_traces_from_excel(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # Coerce all to numeric
    X = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    T = X.shape[1]
    t = np.arange(T, dtype=float)  # <- fixed: 0..T-1

    return X, t


# -----------------------------
# Plot helpers: EXACT style you want
# - All Traces - k (spaghetti)
# - Average Trace - k (thick blue)
# With robust y-limits so it looks like your reference
# -----------------------------
def _robust_ylim(Xc: np.ndarray) -> Tuple[float, float]:
    # robust upper bound (avoid one insane outlier exploding the axis)
    hi = float(np.percentile(Xc, 99.5))
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
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Z-score", fontsize=14)
    ax.set_ylim(y0, y1)
    ax.grid(True, alpha=0.6)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_avg_trace_png(Xc: np.ndarray, t: np.ndarray, out_path: str, title: str) -> None:
    avg = Xc.mean(axis=0)
    y0, y1 = _robust_ylim(Xc)

    fig = plt.figure(figsize=(10, 5.5))
    ax = plt.gca()
    ax.set_facecolor("white")

    ax.plot(t, avg, linewidth=3.2)

    ax.set_title(title, fontsize=20, fontweight="bold", pad=10)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Z-score", fontsize=14)
    ax.set_ylim(y0, y1)
    ax.grid(True, alpha=0.6)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Clustering utilities (mimic cluster_utils.run_clustering)
# - default: K=3 like your reference clusters 1/2/3
# - subclassify: run the same K=3 inside selected cluster with short_prefix
# - output cluster keys like: "1","2","3" then "10","11","12"
# -----------------------------
def _kmeans_cluster(Xc: np.ndarray, k: int = 3, seed: int = 0) -> np.ndarray:
    # if too small, reduce k
    n = Xc.shape[0]
    k_eff = max(1, min(k, n))
    if k_eff == 1:
        return np.zeros(n, dtype=int)

    # KMeans on per-neuron features: use the full trace (works fine)
    km = KMeans(n_clusters=k_eff, random_state=seed, n_init="auto")
    labels = km.fit_predict(Xc)
    return labels.astype(int)


def _make_child_keys(prefix: str, n_children: int) -> List[str]:
    # prefix "" => "1","2","3"
    # prefix "1" => "10","11","12"
    # prefix "12" => "120","121","122"
    if prefix == "":
        return [str(i + 1) for i in range(n_children)]
    return [f"{prefix}{i}" for i in range(n_children)]


def _cluster_and_update_state(
    *,
    X: np.ndarray,
    active_idx: np.ndarray,
    state: Dict[str, Any],
    prefix: str,
    target_idx: Optional[np.ndarray] = None,
    k: int = 3,
) -> Dict[str, Any]:
    """
    If target_idx is None: cluster ALL active neurons into new clusters at prefix root.
    If target_idx is provided: subclassify ONLY those neurons and replace the parent cluster key.
    """

    clusters: Dict[str, List[int]] = state.get("clusters", {})
    # Convert lists -> np arrays when needed
    if target_idx is None:
        # root clustering of active set
        Xc = X[active_idx]
        labels = _kmeans_cluster(Xc, k=k, seed=0)
        keys = _make_child_keys(prefix, int(labels.max()) + 1)

        # reset clusters completely at root
        clusters = {}
        for i, key in enumerate(keys):
            members = active_idx[labels == i].tolist()
            clusters[key] = members

    else:
        # subclassify inside a single cluster (target_idx)
        Xc = X[target_idx]
        labels = _kmeans_cluster(Xc, k=k, seed=0)
        keys = _make_child_keys(prefix, int(labels.max()) + 1)

        # remove the parent cluster key (prefix itself) if it exists
        if prefix in clusters:
            clusters.pop(prefix, None)

        # add children
        for i, key in enumerate(keys):
            members = target_idx[labels == i].tolist()
            clusters[key] = members

    state["clusters"] = clusters
    return state


# -----------------------------
# Export cluster excels
# -----------------------------
def _export_cluster_excels(df_raw: pd.DataFrame, clusters: Dict[str, List[int]], clusters_dir: str, run_id: str) -> Dict[str, str]:
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

        all_path = os.path.join(plots_dir, f"all_traces_{key}.png")
        avg_path = os.path.join(plots_dir, f"avg_trace_{key}.png")

        plot_all_traces_png(Xc, t, all_path, title=f"All Traces - {key}")
        plot_avg_trace_png(Xc, t, avg_path, title=f"Average Trace - {key}")

        out[key] = {
            "all_traces": _artifact_url(run_id, f"plots/all_traces_{key}.png"),
            "avg_trace": _artifact_url(run_id, f"plots/avg_trace_{key}.png"),
        }

    return out


# -----------------------------
# API entrypoints (called by FastAPI)
# -----------------------------
def create_new_run(
    *,
    excel_path: str,
    out_dir: str,
    run_id: str,
    threshold: float,
    k: int = 3,
) -> Dict[str, Any]:
    plots_dir = os.path.join(out_dir, "plots")
    clusters_dir = os.path.join(out_dir, "clusters")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(clusters_dir, exist_ok=True)

    df_raw = pd.read_excel(excel_path)  # expects your z-score matrix
    X, t = extract_traces_from_excel(df_raw)

    # ACTIVE / NON-ACTIVE based on max z-score >= threshold
    activity_score = X.max(axis=1)
    active_mask = activity_score >= float(threshold)
    active_idx = np.where(active_mask)[0]
    non_active_idx = np.where(~active_mask)[0]

    # export active/non-active
    df_raw.iloc[active_idx].to_excel(os.path.join(clusters_dir, "active.xlsx"), index=False)
    df_raw.iloc[non_active_idx].to_excel(os.path.join(clusters_dir, "non_active.xlsx"), index=False)

    # run root clustering on active only (like your flask version)
    state = {"clusters": {}}
    state = _cluster_and_update_state(X=X, active_idx=active_idx, state=state, prefix="", target_idx=None, k=k)

    # exports
    cluster_excels = _export_cluster_excels(df_raw, state["clusters"], clusters_dir, run_id)
    cluster_plots = _write_cluster_plots(X, t, state["clusters"], plots_dir, run_id)

    # persist state for later subclassify
    _save_json(os.path.join(out_dir, "state.json"), {
        "threshold": float(threshold),
        "k": int(k),
        "active_idx": active_idx.tolist(),
        "clusters": state["clusters"],
    })

    clusters_list = [{"id": key, "size": len(idx)} for key, idx in sorted(state["clusters"].items())]

    summary = {
        "run_id": run_id,
        "threshold": float(threshold),
        "counts": {"total": int(len(df_raw)), "active": int(len(active_idx)), "non_active": int(len(non_active_idx))},
        "clusters": clusters_list,
        "artifacts": {
            "active_excel": _artifact_url(run_id, "clusters/active.xlsx"),
            "non_active_excel": _artifact_url(run_id, "clusters/non_active.xlsx"),
            "cluster_excels": cluster_excels,
            "cluster_plots": cluster_plots,
        },
    }

    _save_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def subclassify_selected(
    *,
    out_dir: str,
    run_id: str,
    selected_clusters: List[str],
    k: int = 3,
) -> Dict[str, Any]:
    """
    Mimics your Flask /subclassify:
      selected_keys = request.form.getlist("selected_clusters")
      for key in selected_keys:
        run_clustering(data, short_prefix=key)
    """
    state_path = os.path.join(out_dir, "state.json")
    state = _load_json(state_path, default=None)
    if not state:
        raise RuntimeError("Run state not found. Create a run first.")

    threshold = float(state.get("threshold", 0.0))
    active_idx = np.array(state.get("active_idx", []), dtype=int)
    clusters: Dict[str, List[int]] = state.get("clusters", {})

    # Load original excel again for exports/plots
    excel_path = os.path.join(out_dir, "input.xlsx")
    df_raw = pd.read_excel(excel_path)
    X, t = extract_traces_from_excel(df_raw)

    # For each selected cluster key: subclassify if size>=3
    for key in selected_clusters:
        idx_list = clusters.get(key)
        if not idx_list:
            continue
        if len(idx_list) < 3:
            continue

        target_idx = np.array(idx_list, dtype=int)
        state = _cluster_and_update_state(
            X=X,
            active_idx=active_idx,
            state={"clusters": clusters},
            prefix=key,              # <- short_prefix behavior
            target_idx=target_idx,   # only inside this cluster
            k=k,
        )
        clusters = state["clusters"]

    # write outputs
    plots_dir = os.path.join(out_dir, "plots")
    clusters_dir = os.path.join(out_dir, "clusters")

    cluster_excels = _export_cluster_excels(df_raw, clusters, clusters_dir, run_id)
    cluster_plots = _write_cluster_plots(X, t, clusters, plots_dir, run_id)

    # persist updated state
    _save_json(state_path, {
        "threshold": threshold,
        "k": int(k),
        "active_idx": active_idx.tolist(),
        "clusters": clusters,
    })

    clusters_list = [{"id": key, "size": len(idx)} for key, idx in sorted(clusters.items())]

    summary = {
        "run_id": run_id,
        "threshold": threshold,
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
        },
    }

    _save_json(os.path.join(out_dir, "summary.json"), summary)
    return summary
