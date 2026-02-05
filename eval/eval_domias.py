#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from domias.evaluator import evaluate_performance
from domias.models.generator import GeneratorInterface


class PrecomputedGenerator(GeneratorInterface):
    """
    Wrapper that exposes an already-generated synthetic dataset
    through the DOMIAS GeneratorInterface API.
    """

    def __init__(self, synth_df: pd.DataFrame) -> None:
        # Work on a copy to avoid side-effects
        self.synth_df = synth_df.reset_index(drop=True)

    def fit(self, data: pd.DataFrame) -> "PrecomputedGenerator":
        # No-op: the underlying generative model is assumed pre-trained.
        # We only resample from synth_df in generate().
        return self

    def generate(self, count: int) -> pd.DataFrame:
        # Sample rows from the provided synthetic CSV.
        # Use replacement if the requested count exceeds available rows.
        if count <= len(self.synth_df):
            return self.synth_df.sample(n=count, replace=False, random_state=0).reset_index(drop=True)
        return self.synth_df.sample(n=count, replace=True, random_state=0).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="DOMIAS membership inference evaluation")
    ap.add_argument("--dataname", type=str, default="cardio")
    ap.add_argument("--model", type=str, default="tabddpm",
                    help="Name of synthetic model (used to locate synthetic/{dataname}/{model}.csv)")
    ap.add_argument("--density_estimator", type=str, default="prior",
                    choices=["prior", "kde", "bnaf"],
                    help="Density estimator used inside DOMIAS (Eq. 2).")
    ap.add_argument("--mem_set_size", type=int, default=500,
                    help="Number of member (training) records used in the attack.")
    ap.add_argument("--reference_set_size", type=int, default=5000,
                    help="Number of reference records used to approximate p_R.")
    ap.add_argument("--synthetic_size", type=int, default=10000,
                    help="Number of synthetic samples used inside DOMIAS.")
    ap.add_argument("--training_epochs", type=int, default=2000,
                    help="Epochs parameter passed to DOMIAS' internal CTGAN baseline (LOGAN_0).")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / args.dataname
    syn_path = repo_root / "synthetic" / args.dataname / f"{args.model}.csv"

    if not syn_path.exists():
        raise FileNotFoundError(f"Synthetic file not found: {syn_path}")

    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Expected {train_csv} and {test_csv} to exist.")

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # Convert to numpy arrays (DOMIAS works on numeric arrays)
    train_arr = df_train.to_numpy(dtype=np.float32)
    test_arr = df_test.to_numpy(dtype=np.float32)

    # --- Build member, non-member and reference sets ---
    max_mem = min(args.mem_set_size, train_arr.shape[0])
    max_nonmem = test_arr.shape[0]
    mem_set_size = min(max_mem, max_nonmem)
    if mem_set_size == 0:
        raise ValueError("mem_set_size too large for available train/test sizes.")

    # Members: first mem_set_size train rows
    mem_set = train_arr[:mem_set_size]

    # Non-members: first mem_set_size test rows
    non_mem_set = test_arr[:mem_set_size]

    # Reference candidates: remaining train + remaining test
    ref_candidates = np.vstack([train_arr[mem_set_size:], test_arr[mem_set_size:]])
    if ref_candidates.shape[0] == 0:
        raise ValueError("No data left for reference_set; reduce mem_set_size or reference_set_size.")

    reference_set_size = min(args.reference_set_size, ref_candidates.shape[0])
    reference_set = ref_candidates[-reference_set_size:]

    # DOMIAS expects a single `dataset` array and splits it internally as:
    #   mem_set       = dataset[:mem_set_size]
    #   non_mem_set   = dataset[mem_set_size:2*mem_set_size]
    #   reference_set = dataset[-reference_set_size:]
    dataset_for_domias = np.vstack([mem_set, non_mem_set, reference_set])

    # --- Synthetic data wrapper ---
    synth_df = pd.read_csv(syn_path)
    synthetic_size = min(args.synthetic_size, len(synth_df))
    generator = PrecomputedGenerator(synth_df)

    # Run DOMIAS evaluation (plus baselines) for a single synthetic_size
    perf = evaluate_performance(
        generator=generator,
        dataset=dataset_for_domias,
        mem_set_size=mem_set_size,
        reference_set_size=reference_set_size,
        training_epochs=args.training_epochs,
        synthetic_sizes=[synthetic_size],
        density_estimator=args.density_estimator,
    )

    result_for_size = perf[synthetic_size]
    domias_perf = result_for_size["MIA_performance"]["domias"]

    out = {
        "dataname": args.dataname,
        "model": args.model,
        "mem_set_size": mem_set_size,
        "reference_set_size": reference_set_size,
        "synthetic_size": synthetic_size,
        "density_estimator": args.density_estimator,
        "domias": domias_perf,
        # Optionally, we could expose baseline metrics as well:
        # "all_attacks": result_for_size["MIA_performance"],
    }

    out_dir = repo_root / "eval" / "privacy" / args.dataname
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_domias.json"
    out_path.write_text(json.dumps(out, indent=4) + "\n")

    print(f"Saved DOMIAS results to {out_path}")
    print(
        f"{args.dataname}, {args.model}: "
        f"DOMIAS accuracy={domias_perf['accuracy']:.4f}, "
        f"aucroc={domias_perf['aucroc']:.4f}"
    )


if __name__ == "__main__":
    main()