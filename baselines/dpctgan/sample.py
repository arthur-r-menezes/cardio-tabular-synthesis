# Sample from a trained DP-CTGAN model and save CSV.

import json
import pickle
from pathlib import Path
import pandas as pd

def _safe_import_dp_cgan():
    try:
        from dp_cgans import DP_CGAN
        return DP_CGAN
    except Exception:
        return None  # optional

def _load_info(dataname: str, repo_root: Path):
    primary = repo_root / "data" / dataname / "info.json"
    fallback = repo_root / "data" / "Info" / f"{dataname}.json"
    path = primary if primary.exists() else fallback
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main(args):
    repo_root = Path(__file__).resolve().parents[2]
    dataname = args.dataname

    data_dir = repo_root / "data" / dataname
    train_csv = data_dir / "train.csv"
    info = _load_info(dataname, repo_root)
    column_names = info.get("column_names") or list(pd.read_csv(train_csv).columns)

    num_samples = args.num_samples or int(info.get("train_num", pd.read_csv(train_csv).shape[0]))

    ckpt_dir = repo_root / "ckpt" / "dpctgan"
    model_path = Path(args.load) if getattr(args, "load", None) else (ckpt_dir / f"{dataname}.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"DP-CTGAN checkpoint not found at {model_path}. Train first or pass --load.")

    # Load (prefer classmethod if available)
    DP_CGAN = _safe_import_dp_cgan()
    model = None
    if DP_CGAN is not None:
        try:
            model = DP_CGAN.load(str(model_path))
        except Exception:
            pass
    if model is None:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    sampled = model.sample(num_samples)

    # Reorder columns to match training order if possible
    sampled = sampled.reindex(columns=column_names, fill_value=None)

    save_path = Path(args.save_path) if args.save_path else (repo_root / "synthetic" / dataname / "dpctgan.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(save_path, index=False)
    print(f"Saved {sampled.shape[0]} DP-CTGAN samples to {save_path}")