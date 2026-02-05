#!/usr/bin/env python

# eval/eval_linrecon.py

# This attack is inspired by 2024 Annamalai et al. "A linear Reconstruction Approach for
# Attribute Inference Attacks Against Synthetic Data" https://arxiv.org/pdf/2301.10053

import argparse
import json
from pathlib import Path

import pandas as pd

from linrecon.attack import run_lra


def main() -> None:
    ap = argparse.ArgumentParser(description="Linear Reconstruction Attack (LRA) evaluation")
    ap.add_argument("--dataname", type=str, default="cardio")
    ap.add_argument("--model", type=str, default="tabddpm",
                    help="Name of synthetic model (synthetic/{dataname}/{model}.csv).")
    ap.add_argument("--secret_col", type=str, default=None,
                    help="Name of the binary secret attribute column. "
                         "Default: 'cardio' for cardio; otherwise last column of train.csv.")
    ap.add_argument("--k", type=int, default=3,
                    help="Marginal order k (current implementation supports only k=3).")
    ap.add_argument("--n_queries", type=int, default=2000,
                    help="Maximum number of conditional queries to use in the reconstruction LP.")
    ap.add_argument("--max_records", type=int, default=1000,
                    help="Maximum number of real records included in the reconstruction LP.")
    ap.add_argument("--min_support", type=int, default=5,
                    help="Minimum support in real & synthetic data for a query to be used.")
    ap.add_argument("--random_state", type=int, default=0,
                    help="Random seed for subsampling and query selection.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / args.dataname
    syn_path = repo_root / "synthetic" / args.dataname / f"{args.model}.csv"
    train_csv = data_dir / "train.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Real train.csv not found at {train_csv}")
    if not syn_path.exists():
        raise FileNotFoundError(f"Synthetic CSV not found at {syn_path}")

    df_real = pd.read_csv(train_csv)
    df_syn = pd.read_csv(syn_path)

    # Default secret column
    secret_col = args.secret_col
    if secret_col is None:
        if args.dataname == "cardio" and "cardio" in df_real.columns:
            secret_col = "cardio"
        else:
            secret_col = df_real.columns[-1]

    result = run_lra(
        df_real=df_real,
        df_syn=df_syn,
        secret_col=secret_col,
        k=args.k,
        n_queries=args.n_queries,
        max_records=args.max_records,
        min_support=args.min_support,
        random_state=args.random_state,
    )

    out = {
        "dataname": args.dataname,
        "model": args.model,
        "secret_col": secret_col,
        "k": args.k,
        "n_queries": result["n_queries"],
        "n_records": result["n_records"],
        "accuracy": result["accuracy"],
        "aucroc": result["aucroc"],
    }

    out_dir = repo_root / "eval" / "privacy" / args.dataname
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_linrecon.json"
    out_path.write_text(json.dumps(out, indent=4) + "\n")

    print(
        f"{args.dataname}, {args.model}: "
        f"LRA(secret_col={secret_col}) accuracy={out['accuracy']:.4f}, "
        f"aucroc={out['aucroc']:.4f}"
    )


if __name__ == "__main__":
    main()