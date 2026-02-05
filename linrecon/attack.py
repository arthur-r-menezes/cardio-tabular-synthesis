# linrecon/attack.py
# This attack is inspired by 2024 Annamalai et al. "A linear Reconstruction Approach for
# Attribute Inference Attacks Against Synthetic Data" https://arxiv.org/pdf/2301.10053

from __future__ import annotations

from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.metrics import accuracy_score, roc_auc_score


def _discretize_columns(
    df_real: pd.DataFrame,
    df_syn: pd.DataFrame,
    secret_col: str,
    n_bins: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Discretize all non-secret columns into small integer categories.

    - Numeric columns: binned into `n_bins` quantile bins (if enough unique values),

      otherwise treated as categorical.
    - Non-numeric columns: treated as categorical.

    Missing/out-of-range values are mapped to -1 and later ignored in queries.
    """
    df_real_disc = df_real.copy()
    df_syn_disc = df_syn.copy()

    quasi_cols = [c for c in df_real.columns if c != secret_col]

    for col in quasi_cols:
        s_real = df_real[col]
        s_syn = df_syn[col]

        if pd.api.types.is_numeric_dtype(s_real):
            vals = s_real.dropna().to_numpy()
            unique_vals = np.unique(vals)
            # If few unique values, treat as categorical
            if unique_vals.size <= n_bins:
                categories = sorted(unique_vals.tolist())
                mapping = {v: i for i, v in enumerate(categories)}
                df_real_disc[col] = s_real.map(mapping).astype("float32")
                df_syn_disc[col] = s_syn.map(mapping).astype("float32")
                df_syn_disc[col] = df_syn_disc[col].where(
                    df_syn_disc[col].notna(), -1.0
                )
            else:
                # Quantile-based binning
                qs = np.linspace(0.0, 1.0, n_bins + 1)
                edges = np.unique(np.quantile(vals, qs))
                if edges.size <= 2:
                    # Fallback to categorical
                    categories = sorted(unique_vals.tolist())
                    mapping = {v: i for i, v in enumerate(categories)}
                    df_real_disc[col] = s_real.map(mapping).astype("float32")
                    df_syn_disc[col] = s_syn.map(mapping).astype("float32")
                    df_syn_disc[col] = df_syn_disc[col].where(
                        df_syn_disc[col].notna(), -1.0
                    )
                else:
                    # Use digitize on real and synthetic with same edges
                    def bin_series(series: pd.Series, edges_: np.ndarray) -> np.ndarray:
                        arr = series.to_numpy(dtype="float64")
                        # bins in {0 .. len(edges_)-2}
                        bins = np.digitize(arr, edges_[1:-1], right=False)
                        # mark NaNs as -1
                        bins = np.where(np.isnan(arr), -1, bins)
                        return bins.astype("float32")

                    df_real_disc[col] = bin_series(s_real, edges)
                    df_syn_disc[col] = bin_series(s_syn, edges)
        else:
            # Treat as categorical (string)
            s_real_str = s_real.astype(str)
            categories = sorted(s_real_str.dropna().unique().tolist())
            mapping = {v: i for i, v in enumerate(categories)}
            df_real_disc[col] = s_real_str.map(mapping).astype("float32")
            s_syn_str = s_syn.astype(str)
            df_syn_disc[col] = s_syn_str.map(mapping).astype("float32")
            df_syn_disc[col] = df_syn_disc[col].where(
                df_syn_disc[col].notna(), -1.0
            )

    return df_real_disc, df_syn_disc, quasi_cols


def _build_conditional_queries(
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    n_queries: int,
    min_support: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build up to `n_queries` 3-way conditional queries:

      - 2 quasi-identifier columns (a, b)
      - secret attribute = 1

    Each query j yields:
      - row A_j: indicator over records in X_real that match the (a,b) pair
      - target b_j = p_syn(y=1 | a,b) * (# real records with a,b)

    Only combinations with sufficient support in both real and synthetic data
    are used (>= min_support).
    """
    rng = np.random.default_rng(random_state)
    n, d = X_real.shape

    # possible pairs of quasi-id columns (a, b)
    col_pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    rng.shuffle(col_pairs)

    rows: List[np.ndarray] = []
    bs: List[float] = []

    for a, b in col_pairs:
        vals_r = np.stack([X_real[:, a], X_real[:, b]], axis=1)
        uniq_r, inv_r, counts_r = np.unique(
            vals_r, axis=0, return_inverse=True, return_counts=True
        )

        vals_s = np.stack([X_syn[:, a], X_syn[:, b]], axis=1)
        uniq_s, inv_s, counts_s = np.unique(
            vals_s, axis=0, return_inverse=True, return_counts=True
        )
        syn_map = {
            (int(u[0]), int(u[1])): (int(idx), int(counts_s[idx]))
            for idx, u in enumerate(uniq_s)
        }

        for combo_idx, (va, vb) in enumerate(uniq_r):
            # skip missing / out-of-range
            if va < 0 or vb < 0:
                continue

            N_ab_real = int(counts_r[combo_idx])
            if N_ab_real < min_support:
                continue

            key = (int(va), int(vb))
            if key not in syn_map:
                continue
            syn_idx, N_ab_syn = syn_map[key]
            if N_ab_syn < min_support:
                continue

            mask_syn = (inv_s == syn_idx)
            num_y1_syn = int((y_syn[mask_syn] == 1).sum())
            if N_ab_syn == 0:
                continue

            p_y1_given_ab_syn = num_y1_syn / float(N_ab_syn)
            b_j = p_y1_given_ab_syn * float(N_ab_real)

            mask_real = (inv_r == combo_idx)
            row = mask_real.astype("float32")
            rows.append(row)
            bs.append(b_j)

            if len(rows) >= n_queries:
                break
        if len(rows) >= n_queries:
            break

    if not rows:
        raise RuntimeError(
            "Could not construct any queries for linear reconstruction "
            "(not enough support / overlap between real and synthetic)."
        )

    A = np.stack(rows, axis=0).astype("float64")  # (q, n)
    b = np.array(bs, dtype="float64")             # (q,)
    return A, b


def _solve_linear_program(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the LP:

        min sum_j |e_j|
        s.t.  e_j = b_j - A_j t
              0 <= t_i <= 1

    via SciPy's linprog by splitting e_j into e_j+ and e_j-.
    """
    q, n = A.shape

    # Variables: [t (n), e_plus (q), e_minus (q)]
    c = np.concatenate(
        [
            np.zeros(n, dtype=np.float64),
            np.ones(q, dtype=np.float64),
            np.ones(q, dtype=np.float64),
        ]
    )

    Aeq = np.zeros((q, n + 2 * q), dtype=np.float64)
    Aeq[:, :n] = A
    for j in range(q):
        Aeq[j, n + j] = 1.0          # e_plus_j
        Aeq[j, n + q + j] = -1.0     # e_minus_j

    beq = b

    bounds: List[Tuple[float, float | None]] = []
    # bounds for t: 0 <= t_i <= 1
    bounds.extend((0.0, 1.0) for _ in range(n))
    # bounds for e_plus, e_minus: >= 0
    bounds.extend((0.0, None) for _ in range(2 * q))

    res = linprog(
        c,
        A_eq=Aeq,
        b_eq=beq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Linear program did not converge: {res.message}")

    x = res.x
    t_hat = np.clip(x[:n], 0.0, 1.0)
    return t_hat


def run_lra(
    df_real: pd.DataFrame,
    df_syn: pd.DataFrame,
    secret_col: str,
    k: int = 3,
    n_queries: int = 2000,
    max_records: int = 1000,
    min_support: int = 5,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Run the Linear Reconstruction Attack (Adv_recon-style) in a simplified
    form on a real/synthetic pair.

    - df_real: original (train) data
    - df_syn: synthetic data (same schema)
    - secret_col: binary target attribute (0/1 or two distinct values)
    - k: currently must be 3 (two quasi-id attributes + secret)
    - n_queries: max number of conditional queries to use
    - max_records: max number of real records (subsample for tractable LP)
    - min_support: min support in real & synthetic for a query to be used
    - random_state: RNG seed

    Returns:
        {
          "accuracy": float,
          "aucroc": float,
          "n_records": int,
          "n_queries": int,
          "secret_col": str,
        }
    """
    if k != 3:
        raise ValueError("Current implementation supports only k=3 (two quasi-identifiers + secret).")

    if secret_col not in df_real.columns:
        raise ValueError(f"secret_col '{secret_col}' not found in real data columns.")
    if secret_col not in df_syn.columns:
        raise ValueError(f"secret_col '{secret_col}' not found in synthetic data columns.")

    # Ensure secret is binary and map to {0,1}
    y_real_raw = df_real[secret_col].to_numpy()
    uniq = np.unique(y_real_raw)
    if uniq.size > 2:
        raise ValueError(
            f"Secret column '{secret_col}' has more than 2 unique values; "
            "a binary attribute is required for this LRA implementation."
        )
    if uniq.size < 2:
        raise ValueError(
            f"Secret column '{secret_col}' has only one unique value; "
            "cannot evaluate an attribute inference attack."
        )

    # Map consistently so that higher value becomes 1
    if uniq.size == 2:
        # sort to be reproducible
        uniq_sorted = np.sort(uniq)
        mapping = {uniq_sorted[0]: 0, uniq_sorted[1]: 1}
    else:
        mapping = {uniq[0]: 0, uniq[0]: 1}  # unreachable, but for type checker

    y_real = np.vectorize(mapping.get)(y_real_raw).astype("int32")
    y_syn_raw = df_syn[secret_col].to_numpy()
    y_syn = np.vectorize(mapping.get)(y_syn_raw).astype("int32")

    # Subsample real records to keep LP tractable
    rng = np.random.default_rng(random_state)
    n_total = len(df_real)
    n_use = min(max_records, n_total)
    idx = rng.choice(n_total, size=n_use, replace=False)
    df_real_sub = df_real.iloc[idx].reset_index(drop=True)
    y_real_sub = y_real[idx]

    # Discretize quasi-identifiers
    df_real_disc, df_syn_disc, quasi_cols = _discretize_columns(
        df_real_sub, df_syn, secret_col
    )
    X_real = df_real_disc[quasi_cols].to_numpy(dtype="float32")
    X_syn = df_syn_disc[quasi_cols].to_numpy(dtype="float32")

    # Build queries and solve LP
    A, b = _build_conditional_queries(
        X_real,
        y_real_sub,
        X_syn,
        y_syn,
        n_queries=n_queries,
        min_support=min_support,
        random_state=random_state,
    )
    t_hat = _solve_linear_program(A, b)
    y_hat = (t_hat >= 0.5).astype(int)

    acc = accuracy_score(y_real_sub, y_hat)
    try:
        auc = roc_auc_score(y_real_sub, t_hat)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(acc),
        "aucroc": float(auc),
        "n_records": int(n_use),
        "n_queries": int(A.shape[0]),
        "secret_col": secret_col,
    }