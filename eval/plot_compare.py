#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path
from datetime import datetime

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