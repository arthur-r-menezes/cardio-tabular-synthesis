# Sample from a trained CTGAN and save to args.save_path (default synthetic/{dataname}/ctgan.csv).

import json
from pathlib import Path
import pandas as pd
from ctgan import CTGAN
import torch

import inspect

def _load_ctgan_checkpoint(path: str) -> CTGAN:
    """
    Load a CTGAN checkpoint under PyTorch >= 2.6 by explicitly setting
    weights_only=False. This bypasses the new safe-unpickling restriction,
    which is acceptable here because the checkpoint is produced by our
    own training code (trusted source).
    """
    # Make sure CTGAN class is imported so pickle can find it
    from ctgan import CTGAN as CTGANClass  # noqa: F401

    sig = inspect.signature(torch.load)
    if 'weights_only' in sig.parameters:
        obj = torch.load(path, weights_only=False)
    else:
        obj = torch.load(path)

    if not isinstance(obj, CTGAN):
        raise TypeError(f"Loaded object from {path} is not a CTGAN instance (got {type(obj)})")
    return obj

def _load_info(dataname: str, repo_root: Path):
    info_path_primary = repo_root / "data" / dataname / "info.json"
    info_path_fallback = repo_root / "data" / "Info" / f"{dataname}.json"
    path = info_path_primary if info_path_primary.exists() else info_path_fallback
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _rebuild_sampler_if_needed(ctgan, df, discrete_columns):
    # If the loaded model lacks transformer/data sampler, rebuild them deterministically from train data.
    needs_rebuild = not hasattr(ctgan, "_data_sampler") or not hasattr(ctgan._data_sampler, "_data")
    if needs_rebuild:
        # Ensure discrete columns are object dtype
        for c in discrete_columns:
            if c in df.columns:
                df[c] = df[c].astype("object")
        # Fit transformer on the same training data
        ctgan._transformer.fit(df)  # deterministic
        transformed = ctgan._transformer.transform(df)
        # Recreate data sampler
        from ctgan.data_sampler import DataSampler
        ctgan._data_sampler = DataSampler(transformed, ctgan._transformer, discrete_columns)

def main(args):
    repo_root = Path(__file__).resolve().parents[2]
    dataname = args.dataname

    data_dir = repo_root / "data" / dataname
    train_csv = data_dir / "train.csv"

    info = _load_info(dataname, repo_root)
    column_names = info.get("column_names") or list(pd.read_csv(train_csv).columns)
    cat_idx = info["cat_col_idx"]
    target_idx = info["target_col_idx"]
    task_type = info.get("task_type", "binclass")

    discrete_columns = [column_names[i] for i in cat_idx]
    if task_type == "binclass":
        discrete_columns += [column_names[i] for i in target_idx]

    num_samples = args.num_samples or int(info.get("train_num", pd.read_csv(train_csv).shape[0]))

    ckpt_dir = repo_root / "ckpt" / "ctgan"
    model_path = Path(args.load) if args.load else (ckpt_dir / f"{dataname}.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"CTGAN checkpoint not found at {model_path}. Train first or pass --load.")

    # Load checkpoint manually with weights_only=False (trusted source)
    ctgan = _load_ctgan_checkpoint(str(model_path))

    # Safety: if sampler not present (old pickle), rebuild from train.csv
    df_train = pd.read_csv(train_csv)
    _rebuild_sampler_if_needed(ctgan, df_train, discrete_columns)

    # Optional conditional sampling
    if args.sample_condition_column and args.sample_condition_column_value is not None:
        try:
            sampled = ctgan.sample(num_samples, conditions={args.sample_condition_column: args.sample_condition_column_value})
        except TypeError:
            # Fallback: oversample then filter
            oversampled = ctgan.sample(max(num_samples * 2, 10000))
            sampled = oversampled[oversampled[args.sample_condition_column] == args.sample_condition_column_value]
            if sampled.shape[0] < num_samples:
                sampled = pd.concat([sampled, ctgan.sample(num_samples - sampled.shape[0])], ignore_index=True)
    else:
        sampled = ctgan.sample(num_samples)

    # Reorder to match original training order
    sampled = sampled.reindex(columns=column_names, fill_value=None)

    save_path = Path(args.save_path) if args.save_path else (repo_root / "synthetic" / dataname / "ctgan.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(save_path, index=False)
    print(f"Saved {sampled.shape[0]} CTGAN samples to {save_path}")