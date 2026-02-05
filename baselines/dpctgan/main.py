# Train DP-CTGAN (dp-cgans) on data/{dataname}/train.csv.

import json
import pickle
from pathlib import Path

import pandas as pd
import torch
import os 

def _parse_dims(dims_str: str):
    return tuple(int(x) for x in dims_str.split(",") if x.strip())

def _load_info(dataname: str, repo_root: Path):
    primary = repo_root / "data" / dataname / "info.json"
    fallback = repo_root / "data" / "Info" / f"{dataname}.json"
    path = primary if primary.exists() else fallback
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_import_dp_cgan():
    try:
        from dp_cgans import DP_CGAN
        return DP_CGAN
    except Exception as e:
        raise ImportError(
            "dp-cgans not installed or incompatible. Please run:\n"
            '  pip install "dp-cgans>=0.1.0" "sdv==1.6.0" "rdt==1.9.0"\n'
            f"Original error: {e}"
        )

def main(args):
    DP_CGAN = _safe_import_dp_cgan()

    repo_root = Path(__file__).resolve().parents[2]
    dataname = args.dataname

    # Paths
    data_dir = repo_root / "data" / dataname
    train_csv = data_dir / "train.csv"
    ckpt_dir = repo_root / "baselines" / "dpctgan" / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.save) if getattr(args, "save", None) else (ckpt_dir / f"{dataname}.pkl")
    work_dir = repo_root / "baselines" / "dpctgan" / "work" / dataname
    work_dir.mkdir(parents=True, exist_ok=True)

    # Metadata and discrete columns
    info = _load_info(dataname, repo_root)
    column_names = info.get("column_names") or list(pd.read_csv(train_csv).columns)
    cat_idx = info["cat_col_idx"]
    target_idx = info["target_col_idx"]
    task_type = info.get("task_type", "binclass")

    discrete_columns = [column_names[i] for i in cat_idx]
    if task_type == "binclass":
        discrete_columns += [column_names[i] for i in target_idx]

    df = pd.read_csv(train_csv)

    # Important: cast discrete columns (including target) to object so DP_CGAN treats them as categorical
    for c in discrete_columns:
        if c in df.columns:
            df[c] = df[c].astype("object")

    # Parse dims
    generator_dim = _parse_dims(getattr(args, "generator_dim", "1024,2048,2048,1024"))
    discriminator_dim = _parse_dims(getattr(args, "discriminator_dim", "1024,2048,2048,1024"))

    # DP flags (safe fallbacks if args not present)
    private = bool(getattr(args, "dp_private", True))
    discriminator_steps = int(getattr(args, "discriminator_steps", 1))

    # Construct model
    model = DP_CGAN(
        epochs=int(getattr(args, "epochs", 1000)),
        batch_size=int(getattr(args, "batch_size", 500)),
        log_frequency=True,
        verbose=True,
        generator_dim=generator_dim,
        discriminator_dim=discriminator_dim,
        generator_lr=float(getattr(args, "generator_lr", 2e-4)),
        discriminator_lr=float(getattr(args, "discriminator_lr", 2e-4)),
        discriminator_steps=discriminator_steps,
        private=private,
        # Optional DP knobs if supported by installed version:
        # noise_multiplier=getattr(args, "dp_noise_multiplier", None),
        # max_grad_norm=getattr(args, "dp_max_grad_norm", None),
        # epsilon=getattr(args, "dp_epsilon", None),
        # delta=getattr(args, "dp_delta", None),
    )

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Training DP-CTGAN on {train_csv} | device={device} | private={private}")
    print(f"Discrete columns: {discrete_columns}")
    print(f"Working directory for DP-CTGAN artefacts: {work_dir}")

    # Fit in dedicated work_dir so dp-cgans writes its artefacts there
    prev_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        model.fit(df)
    finally:
        os.chdir(prev_cwd)

    # Save checkpoint (use built-in save if available, else pickle)
    try:
        model.save(str(model_path))
    except Exception:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    print(f"Saved DP-CTGAN model to {model_path}")