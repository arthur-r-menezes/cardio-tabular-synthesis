#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def read_dcr(dataname: str, method: str, repo_root: Path):
    p = repo_root / "eval" / "privacy" / dataname / f"{method}_dcr.txt"
    if not p.exists():
        return None
    try:
        vals = [float(x.strip()) for x in p.read_text().splitlines() if x.strip()]
        return vals[0] if vals else None
    except Exception:
        return None


def read_domias(dataname: str, method: str, repo_root: Path):
    p = repo_root / "eval" / "privacy" / dataname / f"{method}_domias.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text())
        d = obj.get("domias", {})
        return {
            "domias_accuracy": float(d.get("accuracy")),
            "domias_aucroc": float(d.get("aucroc")),
        }
    except Exception:
        return None


def read_linrecon(dataname: str, method: str, repo_root: Path):
    p = repo_root / "eval" / "privacy" / dataname / f"{method}_linrecon.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text())
        return {
            "lra_accuracy": float(obj.get("accuracy")),
            "lra_aucroc": float(obj.get("aucroc")),
            "lra_n_records": int(obj.get("n_records", 0)),
            "lra_n_queries": int(obj.get("n_queries", 0)),
        }
    except Exception:
        return None


def aggregate(dataname: str, methods: list[str], repo_root: Path) -> pd.DataFrame:
    rows = []
    for m in methods:
        row = {"method": m}

        dcr = read_dcr(dataname, m, repo_root)
        if dcr is not None:
            row["dcr_score"] = dcr

        domias = read_domias(dataname, m, repo_root)
        if domias:
            row.update(domias)

        lra = read_linrecon(dataname, m, repo_root)
        if lra:
            row.update(lra)

        rows.append(row)
    return pd.DataFrame(rows)


def plot_privacy(df: pd.DataFrame, dataname: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    sns.set(style="whitegrid")

    # Build a consistent color palette per method (like in plot_compare.py)
    methods = df["method"].tolist()
    base_palette = sns.color_palette("tab10", n_colors=len(methods))
    palette = {m: base_palette[i] for i, m in enumerate(methods)}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # 0) Distance to Closest Record (DCR)
    ax = axes[0]
    if "dcr_score" in df.columns:
        sns.barplot(
            x="method",
            y="dcr_score",
            data=df,
            ax=ax,
            palette=palette,
        )
        ax.set_title("DCR score (closer to 0.5 is better)")
        # Zoom in to [0.75, 0.85]
        ax.set_ylim(0.75, 0.85)
        ax.set_ylabel("DCR score")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    else:
        ax.set_visible(False)

    # 1) DOMIAS AUCROC
    ax = axes[1]
    if "domias_aucroc" in df.columns:
        sns.barplot(
            x="method",
            y="domias_aucroc",
            data=df,
            ax=ax,
            palette=palette,
        )
        ax.set_title("DOMIAS AUCROC (MI attack)")
        # Zoom in to [0.45, 0.55]
        ax.set_ylim(0.45, 0.55)
        ax.set_ylabel("AUCROC")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    else:
        ax.set_visible(False)

    # 2) Linear Reconstruction AUCROC
    ax = axes[2]
    if "lra_aucroc" in df.columns:
        sns.barplot(
            x="method",
            y="lra_aucroc",
            data=df,
            ax=ax,
            palette=palette,
        )
        ax.set_title("Linear Reconstruction AUCROC")
        # Zoom in to [0.45, 0.55]
        ax.set_ylim(0.45, 0.55)
        ax.set_ylabel("AUCROC")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    else:
        ax.set_visible(False)

    # 3) Combined accuracies (DOMIAS vs LRA) with zoom and consistent colors
    ax = axes[3]
    has_domias = "domias_accuracy" in df.columns
    has_lra = "lra_accuracy" in df.columns

    if has_domias or has_lra:
        x = np.arange(len(methods))
        width = 0.35

        # Build values in method order
        if has_domias:
            domias_vals = [
                df.loc[df["method"] == m, "domias_accuracy"].iloc[0]
                for m in methods
            ]
        else:
            domias_vals = None

        if has_lra:
            lra_vals = [
                df.loc[df["method"] == m, "lra_accuracy"].iloc[0]
                for m in methods
            ]
        else:
            lra_vals = None

        colors = [palette[m] for m in methods]

        # Bars: same color per method, different hatch per attack
        if has_domias:
            ax.bar(
                x - width / 2,
                domias_vals,
                width,
                color=colors,
                hatch="//",
                label="DOMIAS acc",
                edgecolor="black",
            )
        if has_lra:
            ax.bar(
                x + width / 2,
                lra_vals,
                width,
                color=colors,
                hatch="\\\\",
                label="LRA acc",
                edgecolor="black",
            )

        ax.set_title("Attack accuracies (membership vs reconstruction)")
        # Zoom in to [0.4, 0.8]
        ax.set_ylim(0.4, 0.8)
        ax.set_ylabel("Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.legend()
    else:
        ax.set_visible(False)

    fig.suptitle(f"Privacy Evaluation — {dataname}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png_path = outdir / f"privacy_compare_{ts}.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    csv_path = outdir / f"privacy_summary_{ts}.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved privacy plots: {png_path}")
    print(f"Saved privacy summary: {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataname", type=str, default="cardio")
    ap.add_argument("--models", type=str, nargs="+",
                    default=["tabddpm", "ctgan", "tabsyn", "stasy"])
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    df = aggregate(args.dataname, args.models, repo)

    outdir = repo / "eval" / "privacy_plots" / args.dataname
    plot_privacy(df, args.dataname, outdir)


if __name__ == "__main__":
    main()