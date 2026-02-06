import torch
import pandas as pd

import os
from pathlib import Path

import argparse
import json

from baselines.great.models.great import GReaT
from baselines.great.models.great_utils import _array_to_dataframe

def _find_latest_checkpoint_dir(ckpt_root: Path) -> Path:
    """
    Find the latest HuggingFace-style checkpoint directory under ckpt_root.

    Looks for subdirectories named 'checkpoint-*' and picks the one with the
    largest numeric suffix (e.g., checkpoint-2000 > checkpoint-1000).

    Raises FileNotFoundError if none exist.
    """
    candidates = [d for d in ckpt_root.glob("checkpoint-*") if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directories found under {ckpt_root}")

    # sort by numeric suffix if possible, else lexicographically
    def _score(p: Path) -> int:
        name = p.name  # e.g., "checkpoint-2000"
        try:
            return int(name.split("-")[-1])
        except Exception:
            return 0

    candidates.sort(key=_score)
    return candidates[-1]

def main(args):

    dataname = args.dataname

    dataset_path = f'data/{dataname}/train.csv'
    info_path = f'data/{dataname}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)
    train_df = pd.read_csv(dataset_path)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_root = Path(curr_dir) / "ckpt" / dataname

    if not ckpt_root.exists():
        raise FileNotFoundError(f"GReaT checkpoint directory not found at {ckpt_root}. Train first.")

    # Initialize GReaT with the same base model and experiment_dir
    great = GReaT(
        "distilgpt2",
        epochs=200,
        save_steps=2000,
        logging_steps=50,
        experiment_dir=str(ckpt_root),
        batch_size=24,
    )

    # Find the latest HF-style checkpoint directory (with config.json, model.safetensors, etc.)
    best_ckpt_dir = _find_latest_checkpoint_dir(ckpt_root)
    print(f"Loading GReaT fine-tuned model from {best_ckpt_dir}")

    # Let GReaT/HF load from that directory
    great.load_finetuned_model(str(best_ckpt_dir))

    # Update GReaT’s internal column info using the original train data
    df = _array_to_dataframe(train_df, columns=None)
    great._update_column_information(df)
    great._update_conditional_information(df, conditional_col=None)

    n_samples = info['train_num']

    samples = great.sample(n_samples, k=100, device=args.device)
    save_path = args.save_path
    samples.to_csv(save_path, index=False)

    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GReaT')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--bs', type=int, default=16, help='(Maximum) batch size')
    args = parser.parse_args()