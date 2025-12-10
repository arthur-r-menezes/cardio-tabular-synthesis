# Train CTGAN on data/{dataname}/train.csv using info.json to determine discrete columns.

import json
from pathlib import Path
import pandas as pd
import torch
from ctgan import CTGAN

def _parse_dims(dims_str: str):
    return tuple(int(x) for x in dims_str.split(",") if x.strip())

def _load_info(dataname: str, repo_root: Path):
    info_path_primary = repo_root / "data" / dataname / "info.json"
    info_path_fallback = repo_root / "data" / "Info" / f"{dataname}.json"
    path = info_path_primary if info_path_primary.exists() else info_path_fallback
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main(args):
    repo_root = Path(__file__).resolve().parents[2]
    dataname = args.dataname

    data_dir = repo_root / "data" / dataname
    train_csv = data_dir / "train.csv"
    ckpt_dir = repo_root / "baselines" / "ctgan" / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.save) if args.save else (ckpt_dir / f"{dataname}.pkl")

    info = _load_info(dataname, repo_root)
    column_names = info.get("column_names") or list(pd.read_csv(train_csv).columns)
    cat_idx = info["cat_col_idx"]
    target_idx = info["target_col_idx"]
    task_type = info.get("task_type", "binclass")

    discrete_columns = [column_names[i] for i in cat_idx]
    if task_type == "binclass":
        discrete_columns += [column_names[i] for i in target_idx]

    df = pd.read_csv(train_csv)
    # Ensure discrete columns are treated as categorical by CTGAN
    for c in discrete_columns:
        if c in df.columns:
            df[c] = df[c].astype("object")

    generator_dim = _parse_dims(args.generator_dim)
    discriminator_dim = _parse_dims(args.discriminator_dim)

    ctgan = CTGAN(
        embedding_dim=args.embedding_dim,
        generator_dim=generator_dim,
        discriminator_dim=discriminator_dim,
        generator_lr=args.generator_lr,
        discriminator_lr=args.discriminator_lr,
        generator_decay=args.generator_decay,
        discriminator_decay=args.discriminator_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=True,
        cuda=torch.cuda.is_available(),
    )

    print(f"Training CTGAN on {train_csv} with discrete columns: {discrete_columns}")
    ctgan.fit(df, discrete_columns)

    # Use CTGAN's native save (persists transformer and sampler)
    ctgan.save(str(model_path))
    print(f"Saved CTGAN model to {model_path}")