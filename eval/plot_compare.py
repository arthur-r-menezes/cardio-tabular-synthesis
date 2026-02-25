#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def read_density(dataname, method, repo):
    base = Path(repo) / "eval" / "density" / dataname / method
    qfile = base / "quality.txt"
    if not qfile.exists():
        return None
    vals = [float(x.strip()) for x in qfile.read_text().splitlines() if x.strip()]
    if len(vals) < 2:
        return None
    return {"shape": vals[0], "trend": vals[1]}

def read_detection(dataname, method, repo):
    p = Path(repo) / "eval" / "detection" / dataname / f"{method}.txt"
    if not p.exists():
        return None
    try:
        return {"logistic_detection": float(p.read_text().strip())}
    except Exception:
        return None
    
def read_pmse(dataname, method, repo):
    p = Path(repo) / "eval" / "pmse" / dataname / f"{method}.txt"
    if not p.exists():
        return None
    try:
        val = float(p.read_text().strip())
        return {"pmse_ratio": val}
    except Exception:
        return None

def read_quality(dataname, method, repo):
    p = Path(repo) / "eval" / "quality" / dataname / f"{method}.txt"
    if not p.exists():
        return None
    vals = [float(x.strip()) for x in p.read_text().splitlines() if x.strip()]
    if len(vals) < 2:
        return None
    return {"alpha_precision": vals[0], "beta_recall": vals[1]}

def read_mle(dataname, method, repo):
    p = Path(repo) / "eval" / "mle" / dataname / f"{method}.json"
    if not p.exists():
        return None
    obj = json.loads(p.read_text())
    # Binclass: prefer weighted_f1 / roc_auc / accuracy from best_* sets
    def pick_metric(group, key, default=None):
        try:
            # group like {"XGBClassifier": {"weighted_f1": ..., ...}}
            model_name = next(iter(group.keys()))
            return float(group[model_name][key])
        except Exception:
            return default
    out = {}
    if "best_weighted_scores" in obj:
        out["mle_weighted_f1"] = pick_metric(obj["best_weighted_scores"], "weighted_f1")
    if "best_auroc_scores" in obj:
        out["mle_roc_auc"] = pick_metric(obj["best_auroc_scores"], "roc_auc")
    if "best_acc_scores" in obj:
        out["mle_accuracy"] = pick_metric(obj["best_acc_scores"], "accuracy")
    # Regression case: r2 and RMSE
    if "best_r2_scores" in obj:
        model_name = next(iter(obj["best_r2_scores"].keys()))
        out["mle_r2"] = float(obj["best_r2_scores"][model_name]["r2"])
    if "best_rmse_scores" in obj:
        model_name = next(iter(obj["best_rmse_scores"].keys()))
        out["mle_rmse"] = float(obj["best_rmse_scores"][model_name]["RMSE"])
    return out or None

def aggregate(dataname, methods, repo):
    rows = []
    for m in methods:
      row = {"method": m}
      for reader in (read_density, read_detection, read_quality, read_mle, read_pmse):
          vals = reader(dataname, m, repo)
          if vals:
              row.update(vals)
      rows.append(row)
    return pd.DataFrame(rows)


def _compute_tsne_for_method(
    dataname: str,
    method: str,
    repo_root: Path,
    max_samples: int = 2000,
    perplexity: float = 30.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Build a joint real+synthetic feature representation and run t-SNE for one method.
    Returns (Z_real, Z_syn) in R^2, or None if data is missing.
    """
    data_dir = repo_root / "data" / dataname
    real_path = repo_root / "synthetic" / dataname / "real.csv"
    syn_path = repo_root / "synthetic" / dataname / f"{method}.csv"

    if not real_path.exists() or not syn_path.exists():
        print(f"[tsne] Skip {method}: missing real or synthetic CSV")
        return None

    with open(data_dir / "info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    real_df = pd.read_csv(real_path)
    syn_df = pd.read_csv(syn_path)

    # Harmonize column indices
    real_df.columns = range(len(real_df.columns))
    syn_df.columns = range(len(syn_df.columns))

    num_idx = list(info["num_col_idx"])
    cat_idx = list(info["cat_col_idx"])
    target_idx = list(info["target_col_idx"])
    task_type = info["task_type"]

    # For binclass: target treated as categorical; for regression: as numeric
    if task_type == "regression":
        num_idx = num_idx + target_idx
    else:
        cat_idx = cat_idx + target_idx

    # Extract numerical / categorical parts
    n_real = real_df.shape[0]
    n_syn = syn_df.shape[0]

    real_num = (
        real_df[num_idx].to_numpy(dtype=np.float32) if num_idx else np.empty((n_real, 0), dtype=np.float32)
    )
    syn_num = (
        syn_df[num_idx].to_numpy(dtype=np.float32) if num_idx else np.empty((n_syn, 0), dtype=np.float32)
    )

    real_cat = (
        real_df[cat_idx].astype(str).to_numpy() if cat_idx else np.empty((n_real, 0), dtype=str)
    )
    syn_cat = (
        syn_df[cat_idx].astype(str).to_numpy() if cat_idx else np.empty((n_syn, 0), dtype=str)
    )

    # Numeric scaling
    if real_num.shape[1] > 0:
        scaler = StandardScaler()
        num_all = np.vstack([real_num, syn_num])
        num_all_scaled = scaler.fit_transform(num_all)
    else:
        num_all_scaled = np.empty((n_real + n_syn, 0), dtype=np.float32)

    # One-hot encoding for categoricals
    if real_cat.shape[1] > 0:
        try:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            # Older sklearn
            ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        cat_all = np.vstack([real_cat, syn_cat])
        cat_all_oh = ohe.fit_transform(cat_all)
    else:
        cat_all_oh = np.empty((n_real + n_syn, 0), dtype=np.float32)

    X_all = np.hstack([num_all_scaled, cat_all_oh]).astype(np.float32)

    if X_all.shape[1] == 0:
        print(f"[tsne] Skip {method}: empty feature matrix")
        return None

    # Subsample for t-SNE
    max_per = max(1, max_samples)
    rng = np.random.default_rng(0)

    idx_real = np.arange(n_real)
    if n_real > max_per:
        idx_real = rng.choice(idx_real, size=max_per, replace=False)

    idx_syn = np.arange(n_syn)
    if n_syn > max_per:
        idx_syn = rng.choice(idx_syn, size=max_per, replace=False)

    X_real = X_all[idx_real]
    X_syn = X_all[n_real + idx_syn]

    X = np.vstack([X_real, X_syn])
    labels = np.concatenate(
        [np.zeros(len(idx_real), dtype=int), np.ones(len(idx_syn), dtype=int)]
    )

    # Perplexity must be < number of samples
    eff_perp = min(perplexity, max(5.0, (X.shape[0] - 1) / 3.0))

    tsne = TSNE(
        n_components=2,
        perplexity=eff_perp,
        learning_rate="auto",
        init="random",
        random_state=0,
    )
    Z = tsne.fit_transform(X)

    Z_real = Z[labels == 0]
    Z_syn = Z[labels == 1]
    return Z_real, Z_syn


def plot_tsne_grid(
    dataname: str,
    methods: list[str],
    repo_root: Path,
    outdir: Path,
    palette: dict[str, any],
    max_samples: int = 2000,
    perplexity: float = 30.0,
) -> None:
    """
    For each method, compute a t-SNE embedding (real + synthetic)
    and plot them in a single grid:
      - real points: grey
      - synthetic points: method-specific color (from palette)

    """
    results: list[tuple[str, np.ndarray, np.ndarray]] = []
    for m in methods:
        res = _compute_tsne_for_method(
            dataname=dataname,
            method=m,
            repo_root=repo_root,
            max_samples=max_samples,
            perplexity=perplexity,
        )
        if res is None:
            continue
        Z_real, Z_syn = res
        results.append((m, Z_real, Z_syn))

    if not results:
        print("[tsne] No t-SNE plots generated (no valid methods).")
        return

    n = len(results)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols + 2, 4 * nrows),
        squeeze=False,
    )

    for idx, (method, Z_real, Z_syn) in enumerate(results):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        # Real points in grey
        ax.scatter(
            Z_real[:, 0],
            Z_real[:, 1],
            s=5,
            alpha=0.3,
            color="gray",
            label="Real" if idx == 0 else None,
        )
        # Synthetic points in method color
        ax.scatter(
            Z_syn[:, 0],
            Z_syn[:, 1],
            s=5,
            alpha=0.5,
            color=palette.get(method, "C0"),
            label=method if idx == 0 else None,
        )
        ax.set_title(method)
        ax.set_xticks([])
        ax.set_yticks([])

        if idx == 0:
            ax.legend(loc="best", fontsize=8)

    # Hide unused axes
    for idx in range(len(results), nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].set_visible(False)

    fig.suptitle(f"t-SNE: Real vs Synthetic — {dataname}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = outdir / f"tsne_grid_{ts}.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    print(f"Saved t-SNE grid: {png_path}")

def _plot_metric_grid(
    df: pd.DataFrame,
    dataname: str,
    outdir: Path,
    group_name: str,
    file_prefix: str,
    metrics: list[tuple[str, str]],
    ncols: int,
    ts: str,
    palette=None,
    y_lim: tuple[float, float] | None = None,
    log_metrics: set[str] | None = None,
) -> None:
    """
    Helper to plot a horizontal grid of bar charts, one metric per axis.

    metrics: list of (column_name, pretty_label)
    ncols: number of columns (axes) in the grid
    palette: dict mapping method -> color (shared across all plots)
    file_prefix: safe string (no slashes) used for the output filename
    y_lim: optional (ymin, ymax) to set on all axes
    log_metrics: optional set of metric names to plot on a log y-axis
    """
    # Filter metrics that actually exist in df
    available = [(m, label) for (m, label) in metrics if m in df.columns]
    if not available:
        return

    n_plots = len(available)
    ncols = min(ncols, n_plots)
    nrows = int(np.ceil(n_plots / ncols))

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols + 2, 3.5 * nrows),
        squeeze=False,
    )

    for idx, (metric, label) in enumerate(available):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        sns.barplot(
            x="method",
            y=metric,
            data=df,
            ax=ax,
            palette=palette,
        )
        ax.set_title(label)
        ax.set_xlabel("Method")
        ax.set_ylabel(metric)

        # Optional y-axis zoom (e.g., MLE)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        # Optional log scale for selected metrics (e.g., pmse_ratio)
        if log_metrics is not None and metric in log_metrics:
            # pMSE ratio should be > 0; if any non-positive slip through, clip them
            ymin, ymax = ax.get_ylim()
            if ymin <= 0:
                ymin = min(v for v in df[metric] if v > 0) * 0.5
                ax.set_ylim(bottom=ymin, top=ymax)
            ax.set_yscale("log")
            # Null baseline at ratio = 1
            ax.axhline(
                1.0,
                color="gray",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
            )

        # Rotate x tick labels
        ax.tick_params(axis="x", labelrotation=30)
        for tick_label in ax.get_xticklabels():
            tick_label.set_horizontalalignment("right")

    # Hide unused axes
    for idx in range(len(available), nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].set_visible(False)

    fig.suptitle(f"{group_name} — {dataname}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fname = f"{file_prefix}_{ts}.png"
    png_path = outdir / fname
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    print(f"Saved {group_name} plot: {png_path}")

def plot_all(df, dataname, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Consistent color palette per method
    methods = df["method"].unique().tolist()
    base_palette = sns.color_palette("tab10", n_colors=len(methods))
    palette = {m: base_palette[i] for i, m in enumerate(methods)}

    # 1) SDMetrics Density: Shape & Trend (1x2)
    _plot_metric_grid(
        df=df,
        dataname=dataname,
        outdir=outdir,
        group_name="SDMetrics Density",
        file_prefix="sdmetrics_density",
        metrics=[
            ("shape", "Shape"),
            ("trend", "Trend"),
        ],
        ncols=2,
        ts=ts,
        palette=palette,
    )

    # 2) Detection / Two-sample metrics (1x2), log-y + baseline for pMSE ratio
    _plot_metric_grid(
        df=df,
        dataname=dataname,
        outdir=outdir,
        group_name="Detection / Two-sample",
        file_prefix="detection_two_sample",
        metrics=[
            ("logistic_detection", "Logistic Detection"),
            ("pmse_ratio", "pMSE Ratio"),
        ],
        ncols=2,
        ts=ts,
        palette=palette,
        log_metrics={"pmse_ratio"},
    )

    # 3) SynthCity: Alpha Precision & Beta Recall (1x2)
    _plot_metric_grid(
        df=df,
        dataname=dataname,
        outdir=outdir,
        group_name="SynthCity",
        file_prefix="synthcity",
        metrics=[
            ("alpha_precision", "Alpha Precision"),
            ("beta_recall", "Beta Recall"),
        ],
        ncols=2,
        ts=ts,
        palette=palette,
    )

    # 4) MLE downstream scores (1x3), zoom y-axis to [0.6, 0.8]
    _plot_metric_grid(
        df=df,
        dataname=dataname,
        outdir=outdir,
        group_name="MLE Downstream Scores",
        file_prefix="mle_downstream_scores",
        metrics=[
            ("mle_weighted_f1", "Weighted F1"),
            ("mle_roc_auc", "ROC AUC"),
            ("mle_accuracy", "Accuracy"),
        ],
        ncols=3,
        ts=ts,
        palette=palette,
        y_lim=(0.6, 0.8),
    )

    # 5) t-SNE grid: real (grey) vs synthetic (method color)
    repo_root = Path(__file__).resolve().parents[1]
    plot_tsne_grid(
        dataname=dataname,
        methods=methods,
        repo_root=repo_root,
        outdir=outdir,
        palette=palette,
        max_samples=2000,
        perplexity=30.0,
    )

    # Summary CSV
    csv_path = outdir / f"summary_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary: {csv_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataname", type=str, default="cardio")
    ap.add_argument("--models", type=str, nargs="+", default=["tabddpm","ctgan","dpctgan","tabsyn","great","stasy"])
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]  # repo root
    df = aggregate(args.dataname, args.models, repo)
    outdir = repo / "eval" / "plots" / args.dataname
    plot_all(df, args.dataname, outdir)

if __name__ == "__main__":
    main()