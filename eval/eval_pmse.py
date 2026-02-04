# eval/eval_pmse.py

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='adult')
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--path', type=str, default=None, help='Optional explicit path to synthetic CSV')
    args = parser.parse_args()

    dataname = args.dataname
    model = args.model

    if args.path:
        syn_path = args.path
    else:
        syn_path = f'synthetic/{dataname}/{model}.csv'
    real_path = f'synthetic/{dataname}/real.csv'
    data_dir = f'data/{dataname}'

    if not os.path.exists(syn_path):
        raise FileNotFoundError(f"Synthetic file not found: {syn_path}")
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Real file not found: {real_path}")

    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    syn_df = pd.read_csv(syn_path)
    real_df = pd.read_csv(real_path)

    # Harmonize column indices
    real_df.columns = range(len(real_df.columns))
    syn_df.columns = range(len(syn_df.columns))

    num_idx = list(info['num_col_idx'])
    cat_idx = list(info['cat_col_idx'])
    target_idx = list(info['target_col_idx'])
    task_type = info['task_type']

    # Für binäre/multiclass Klassifikation: Target als kategorial behandeln
    if task_type == 'regression':
        num_idx = num_idx + target_idx
    else:
        cat_idx = cat_idx + target_idx

    # Numerische und kategoriale Teile extrahieren
    real_num = real_df[num_idx].to_numpy(dtype=np.float32) if num_idx else np.empty((real_df.shape[0], 0))
    syn_num = syn_df[num_idx].to_numpy(dtype=np.float32) if num_idx else np.empty((syn_df.shape[0], 0))

    real_cat = real_df[cat_idx].astype(str).to_numpy() if cat_idx else np.empty((real_df.shape[0], 0), dtype=str)
    syn_cat = syn_df[cat_idx].astype(str).to_numpy() if cat_idx else np.empty((syn_df.shape[0], 0), dtype=str)

    n_real = real_df.shape[0]
    n_syn = syn_df.shape[0]
    N = n_real + n_syn

    # Numerische Skalierung
    if real_num.shape[1] > 0:
        scaler = StandardScaler()
        num_all = np.vstack([real_num, syn_num])
        num_all_scaled = scaler.fit_transform(num_all)
    else:
        num_all_scaled = np.empty((N, 0), dtype=np.float32)

    # One-Hot-Encoding kategorialer Features
    if real_cat.shape[1] > 0:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        cat_all = np.vstack([real_cat, syn_cat])
        cat_all_oh = ohe.fit_transform(cat_all)
    else:
        cat_all_oh = np.empty((N, 0), dtype=np.float32)

    X = np.hstack([num_all_scaled, cat_all_oh]).astype(np.float32)
    y = np.concatenate([np.zeros(n_real, dtype=int), np.ones(n_syn, dtype=int)])

    # Wenn keine Features vorhanden sind, kann pMSE nicht sinnvoll berechnet werden
    if X.shape[1] == 0:
        raise RuntimeError("No features available for pMSE computation (empty design matrix).")

    # Logistische Regression (Diskriminator)
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(X, y)
    s = clf.predict_proba(X)[:, 1]

    # Observed Utility
    p_syn = n_syn / N
    observed = np.mean((s - p_syn) ** 2)

    # Expected Utility gemäß Snoke et al.: p*(1-p)*d/N
    p_real = n_real / N
    d = X.shape[1]
    expected = p_real * (1.0 - p_real) * d / N if N > 0 else np.nan

    ratio = observed / expected if expected > 0 else np.nan

    save_dir = f'eval/pmse/{dataname}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{model}.txt'
    with open(save_path, 'w') as f:
        f.write(f"{ratio}\n")

    print(f"{dataname}, {model}: pMSE_ratio={ratio}")


if __name__ == '__main__':
    main()