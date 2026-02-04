# eval/eval_tsne.py

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='adult')
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--path', type=str, default=None, help='Optional explicit path to synthetic CSV')
    parser.add_argument('--max_samples', type=int, default=2000, help='Max samples per source (real/syn) for t-SNE')
    parser.add_argument('--perplexity', type=float, default=30.0, help='t-SNE perplexity')
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

    real_df.columns = range(len(real_df.columns))
    syn_df.columns = range(len(syn_df.columns))

    num_idx = list(info['num_col_idx'])
    cat_idx = list(info['cat_col_idx'])
    target_idx = list(info['target_col_idx'])
    task_type = info['task_type']

    if task_type == 'regression':
        num_idx = num_idx + target_idx
    else:
        cat_idx = cat_idx + target_idx

    real_num = real_df[num_idx].to_numpy(dtype=np.float32) if num_idx else np.empty((real_df.shape[0], 0))
    syn_num = syn_df[num_idx].to_numpy(dtype=np.float32) if num_idx else np.empty((syn_df.shape[0], 0))

    real_cat = real_df[cat_idx].astype(str).to_numpy() if cat_idx else np.empty((real_df.shape[0], 0), dtype=str)
    syn_cat = syn_df[cat_idx].astype(str).to_numpy() if cat_idx else np.empty((syn_df.shape[0], 0), dtype=str)

    # Numerische Skalierung
    if real_num.shape[1] > 0:
        scaler = StandardScaler()
        num_all = np.vstack([real_num, syn_num])
        num_all_scaled = scaler.fit_transform(num_all)
    else:
        num_all_scaled = np.empty((real_df.shape[0] + syn_df.shape[0], 0), dtype=np.float32)

    # One-Hot-Encoding kategorialer Features
    if real_cat.shape[1] > 0:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        cat_all = np.vstack([real_cat, syn_cat])
        cat_all_oh = ohe.fit_transform(cat_all)
    else:
        cat_all_oh = np.empty((real_df.shape[0] + syn_df.shape[0], 0), dtype=np.float32)

    X_all = np.hstack([num_all_scaled, cat_all_oh]).astype(np.float32)
    n_real = real_df.shape[0]
    n_syn = syn_df.shape[0]

    # Subsampling für t-SNE
    max_per = max(1, args.max_samples)
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
    labels = np.concatenate([np.zeros(len(idx_real), dtype=int), np.ones(len(idx_syn), dtype=int)])

    if X.shape[1] == 0:
        raise RuntimeError("No features available for t-SNE (empty design matrix).")

    tsne = TSNE(
        n_components=2,
        perplexity=min(args.perplexity, max(5.0, (X.shape[0] - 1) / 3.0)),
        learning_rate='auto',
        init='random',
        random_state=0,
    )
    Z = tsne.fit_transform(X)

    save_dir = f'eval/tsne/{dataname}'
    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, f'{model}.png')

    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 5))
    plt.scatter(Z[labels == 0, 0], Z[labels == 0, 1], s=8, alpha=0.5, label='real')
    plt.scatter(Z[labels == 1, 0], Z[labels == 1, 1], s=8, alpha=0.5, label='synthetic')
    plt.title(f't-SNE projection: {dataname} / {model}')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"Saved t-SNE plot to {png_path}")


if __name__ == '__main__':
    main()