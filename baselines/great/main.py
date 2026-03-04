# baselines/great/main.py

import os
import argparse
from pathlib import Path

import pandas as pd

from baselines.great.models.great import GReaT


def _find_latest_checkpoint_dir(ckpt_root: Path) -> str | None:
    """
    Find the latest HuggingFace-style checkpoint directory under ckpt_root.

    Looks for subdirs named 'checkpoint-*' and picks the one with the
    largest numeric suffix. Returns its string path, or None if none exist.
    """
    candidates = [d for d in ckpt_root.glob("checkpoint-*") if d.is_dir()]
    if not candidates:
        return None

    def _score(p: Path) -> int:
        name = p.name  # e.g. "checkpoint-150000"
        try:
            return int(name.split("-")[-1])
        except Exception:
            return -1

    candidates.sort(key=_score)
    return str(candidates[-1])


def main(args):
    dataname = args.dataname
    batch_size = args.bs
    dataset_path = f"data/{dataname}/train.csv"
    train_df = pd.read_csv(dataset_path)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_root = Path(curr_dir) / "ckpt" / dataname
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # Auto-detect last checkpoint (if any)
    resume_from_checkpoint = _find_latest_checkpoint_dir(ckpt_root)
    if resume_from_checkpoint:
        print(f"[GReaT] Resuming from checkpoint: {resume_from_checkpoint}")
    else:
        print("[GReaT] No existing checkpoint found; starting from scratch.")

    great = GReaT(
        "distilgpt2",
        epochs=100,
        save_steps=50000,          # fewer checkpoints to reduce disk usage
        logging_steps=50,
        experiment_dir=str(ckpt_root),
        batch_size=batch_size,
    )

    # Pass resume_from_checkpoint into the internal HF Trainer
    trainer = great.fit(train_df, resume_from_checkpoint=resume_from_checkpoint)

    # Save final model artifacts into ckpt_root
    great.save(str(ckpt_root))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GReaT")
    parser.add_argument("--dataname", type=str, default="adult", help="Name of dataset.")
    parser.add_argument("--bs", type=int, default=16, help="(Maximum) batch size")
    args = parser.parse_args()
    main(args)