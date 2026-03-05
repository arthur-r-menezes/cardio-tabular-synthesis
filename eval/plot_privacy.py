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

    # Consistent color palette per method (like utility plots)
    methods = df["method"].tolist()
    base_palette = sns.color_palette("tab10", n_colors=len(methods))
    palette = {m: base_palette[i] for i, m in enumerate(methods)}

    # ------------------------------------------------------------------
    # Image 1: DCR Score (single plot)
    # ------------------------------------------------------------------
    if "dcr_score" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))

        sns.barplot(
            x="method",
            y="dcr_score",
            data=df,
            ax=ax,
            palette=palette,
        )
        ax.set_title(f"DCR score — {dataname} (closer to 0.5 is better)")
        # Cardio: ~0.898–0.903
        ax.set_ylim(0.897, 0.904)
        ax.set_ylabel("DCR score")
        ax.set_xlabel("Method")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

        png_path = outdir / f"privacy_dcr_{ts}.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=200)
        plt.close(fig)
        print(f"Saved DCR plot: {png_path}")
    else:
        print("[privacy] No 'dcr_score' column found; skipping DCR plot.")

    # ------------------------------------------------------------------
    # Image 2: DOMIAS (2x1: AUCROC + accuracy)
    # ------------------------------------------------------------------
    has_domias_auc = "domias_aucroc" in df.columns
    has_domias_acc = "domias_accuracy" in df.columns

    if has_domias_auc or has_domias_acc:
        fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        ax_auc, ax_acc = axes

        if has_domias_auc:
            sns.barplot(
                x="method",
                y="domias_aucroc",
                data=df,
                ax=ax_auc,
                palette=palette,
            )
            ax_auc.set_title("DOMIAS AUCROC (MI attack)")
            # Cardio: ~0.501–0.506
            ax_auc.set_ylim(0.500, 0.508)
            ax_auc.set_ylabel("AUCROC")
        else:
            ax_auc.set_visible(False)

        if has_domias_acc:
            sns.barplot(
                x="method",
                y="domias_accuracy",
                data=df,
                ax=ax_acc,
                palette=palette,
            )
            ax_acc.set_title("DOMIAS accuracy")
            # Use your original attack-accuracy zoom
            ax_acc.set_ylim(0.40, 0.80)
            ax_acc.set_ylabel("Accuracy")
        else:
            ax_acc.set_visible(False)

        # Shared x-axis formatting
        if has_domias_auc or has_domias_acc:
            ax_acc.set_xlabel("Method")
            for ax in axes:
                if ax.get_visible():
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

        fig.suptitle(f"DOMIAS — {dataname}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        png_path = outdir / f"privacy_domias_{ts}.png"
        fig.savefig(png_path, dpi=200)
        plt.close(fig)
        print(f"Saved DOMIAS plot: {png_path}")
    else:
        print("[privacy] No DOMIAS columns found; skipping DOMIAS plots.")

    # ------------------------------------------------------------------
    # Image 3: Linear Reconstruction (2x1: AUCROC + accuracy)
    # ------------------------------------------------------------------
    has_lra_auc = "lra_aucroc" in df.columns
    has_lra_acc = "lra_accuracy" in df.columns

    if has_lra_auc or has_lra_acc:
        fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        ax_auc, ax_acc = axes

        if has_lra_auc:
            sns.barplot(
                x="method",
                y="lra_aucroc",
                data=df,
                ax=ax_auc,
                palette=palette,
            )
            ax_auc.set_title("Linear Reconstruction AUCROC")
            # Cardio: ~0.604–0.699
            ax_auc.set_ylim(0.60, 0.71)
            ax_auc.set_ylabel("AUCROC")
        else:
            ax_auc.set_visible(False)

        if has_lra_acc:
            sns.barplot(
                x="method",
                y="lra_accuracy",
                data=df,
                ax=ax_acc,
                palette=palette,
            )
            ax_acc.set_title("Linear Reconstruction accuracy")
            ax_acc.set_ylim(0.40, 0.80)
            ax_acc.set_ylabel("Accuracy")
        else:
            ax_acc.set_visible(False)

        if has_lra_auc or has_lra_acc:
            ax_acc.set_xlabel("Method")
            for ax in axes:
                if ax.get_visible():
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

        fig.suptitle(f"Linear Reconstruction — {dataname}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        png_path = outdir / f"privacy_lra_{ts}.png"
        fig.savefig(png_path, dpi=200)
        plt.close(fig)
        print(f"Saved LRA plot: {png_path}")
    else:
        print("[privacy] No LRA columns found; skipping LRA plots.")

    # ------------------------------------------------------------------
    # CSV summary (unchanged)
    # ------------------------------------------------------------------
    csv_path = outdir / f"privacy_summary_{ts}.csv"
    df.to_csv(csv_path, index=False)
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