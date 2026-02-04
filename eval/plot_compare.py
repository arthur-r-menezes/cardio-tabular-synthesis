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


def plot_all(df, dataname, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)

    # Dichte: shape/trend (unverändert)
    ax1 = fig.add_subplot(gs[0, 0])
    for metric in ["shape", "trend"]:
        if metric in df.columns:
            sns.barplot(x="method", y=metric, data=df, ax=ax1, label=metric, alpha=0.7)
    ax1.set_title("SDMetrics Density: Shape & Trend")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right")
    ax1.legend()

    # Detection + pMSE-Ratio
    ax2 = fig.add_subplot(gs[0, 1])
    plotted = False
    for metric in ["logistic_detection", "pmse_ratio"]:
        if metric in df.columns:
            sns.barplot(x="method", y=metric, data=df, ax=ax2, label=metric, alpha=0.7)
            plotted = True
    ax2.set_title("Detection / Two-sample metrics")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right")
    if plotted:
        ax2.legend()

    # Qualität: alpha/beta (unverändert)
    ax3 = fig.add_subplot(gs[1, 0])
    for metric in ["alpha_precision", "beta_recall"]:
        if metric in df.columns:
            sns.barplot(x="method", y=metric, data=df, ax=ax3, label=metric, alpha=0.7)
    ax3.set_title("SynthCity Alpha Precision / Beta Recall")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, ha="right")
    ax3.legend()

    # MLE-Downstream-Scores (unverändert außer evtl. import-Updates)
    ax4 = fig.add_subplot(gs[1, 1])
    plotted = False
    for metric in ["mle_weighted_f1", "mle_roc_auc", "mle_accuracy", "mle_r2"]:
        if metric in df.columns:
            sns.barplot(x="method", y=metric, data=df, ax=ax4, label=metric, alpha=0.7)
            plotted = True
    if "mle_rmse" in df.columns:
        df["_mle_rmse_neg"] = -df["mle_rmse"]
        sns.barplot(x="method", y="_mle_rmse_neg", data=df, ax=ax4, label="(-) rmse", alpha=0.7)
        plotted = True
    ax4.set_title("MLE downstream scores")
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=30, ha="right")
    if plotted:
        ax4.legend()

    fig.suptitle(f"Comparative Evaluation — {dataname}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png_path = outdir / f"compare_{ts}.png"
    fig.savefig(png_path, dpi=200)
    csv_path = outdir / f"summary_{ts}.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved plots: {png_path}")
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