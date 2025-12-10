# download_cardio_dataset.py

# Downloads and prepares the Kaggle cardiovascular disease dataset.

# - Creates data/cardio/{train.csv, test.csv}

# - Saves data/cardio/{X_num_train.npy, X_cat_train.npy, y_train.npy, X_num_test.npy, X_cat_test.npy, y_test.npy}

# - Writes data/cardio/info.json (and data/Info/cardio.json) with metadata

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import kagglehub
except ImportError:
    print("kagglehub is not installed. Please run: pip install kagglehub")
    sys.exit(1)

# Kaggle dataset

DATASET_SLUG = "sulianova/cardiovascular-disease-dataset"
DATASET_NAME = "cardio"

# Paths

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data" / DATASET_NAME
INFO_DIR = REPO_ROOT / "data" / "Info"
INFO_JSON_CARDIO = DATA_DIR / "info.json"
INFO_JSON_GLOBAL = INFO_DIR / f"{DATASET_NAME}.json"

# Output CSVs

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

# Synthetic copies (for evaluation scripts that expect them)

SYN_DIR = REPO_ROOT / "synthetic" / DATASET_NAME
SYN_REAL_CSV = SYN_DIR / "real.csv"
SYN_TEST_CSV = SYN_DIR / "test.csv"

# Train/test split ratio (align with preprocess_dataset.py: 90/10)

TEST_RATIO = 0.10
RANDOM_STATE = 1234

# Dataset columns

NUM_COLUMNS = ["age", "height", "weight", "ap_hi", "ap_lo"]
CAT_COLUMNS = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
TARGET_COLUMN = "cardio"


def list_candidate_csvs(download_path: Path):
    all_csv = list(download_path.glob("**/*.csv"))
    if not all_csv:
        raise FileNotFoundError(f"No CSV files found under {download_path}")

    train_first = [p for p in all_csv if "train" in p.name.lower()]
    cardio_main = [p for p in all_csv if p.name.lower() == "cardio.csv"]
    non_test = [p for p in all_csv if "test" not in p.name.lower()]

    ordered = []
    ordered += train_first
    ordered += cardio_main
    ordered += [p for p in non_test if p not in train_first and p not in cardio_main]
    ordered += [p for p in all_csv if p not in ordered]

    seen = set()
    result = []
    for p in ordered:
        if p not in seen:
            result.append(p)
            seen.add(p)
    return result


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().replace(" ", "") for c in df.columns]
    final_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "aphi":
            final_map[c] = "ap_hi"
        elif cl in ("aplo", "ap_lo"):
            final_map[c] = "ap_lo"
        elif cl in (
            "cardio",
            "gender",
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "age",
            "height",
            "weight",
            "id",
        ):
            final_map[c] = cl  # standardize to lowercase names
    if final_map:
        df = df.rename(columns=final_map)
    return df


def read_csv_robust(csv_path: Path) -> pd.DataFrame:
    # Explicit semicolon and UTF-8 BOM handling
    df = pd.read_csv(csv_path, sep=";", engine="python", encoding="utf-8-sig")
    # If not parsed correctly (single header cell with semicolons), fix manually
    if len(df.columns) == 1 and ";" in df.columns[0]:
        raw = csv_path.read_text(encoding="utf-8-sig").splitlines()
        rows = [r.split(";") for r in raw]
        header = [h.strip() for h in rows[0]]
        data = [[v.strip() for v in r] for r in rows[1:]]
        df = pd.DataFrame(data, columns=header)
    # Drop unnamed/index-like columns
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = normalize_columns(df)
    return df


def find_source_csv(download_path: Path) -> Path:
    candidates = list_candidate_csvs(download_path)
    return candidates[0]


def ensure_categorical_coverage_split(df: pd.DataFrame, cat_columns, test_ratio, seed_start=RANDOM_STATE):
    total_num = df.shape[0]
    num_train = int(total_num * (1 - test_ratio))
    num_test = total_num - num_train

    seed = seed_start
    idx = np.arange(total_num)

    while True:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Ensure train has all categories present in the full dataset
        ok = True
        for col in cat_columns:
            if len(set(train_df[col])) != len(set(df[col])):
                ok = False
                break

        if ok:
            return train_df.copy(), test_df.copy(), seed
        seed += 1


def build_mappings(column_names, num_col_idx, cat_col_idx, target_col_idx):
    # Map original indices to grouped indices: num -> [0..], cat -> [n..], target -> [..]
    idx_mapping = {}
    inverse_idx_mapping = {}
    idx_name_mapping = {}

    curr_num = 0
    curr_cat = len(num_col_idx)
    curr_target = curr_cat + len(cat_col_idx)

    for i in range(len(column_names)):
        if i in num_col_idx:
            idx_mapping[int(i)] = curr_num
            curr_num += 1
        elif i in cat_col_idx:
            idx_mapping[int(i)] = curr_cat
            curr_cat += 1
        else:
            idx_mapping[int(i)] = curr_target
            curr_target += 1

    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = int(k)

    for i, name in enumerate(column_names):
        idx_name_mapping[int(i)] = name

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def build_metadata_info(name, task_type, column_names, num_col_idx, cat_col_idx, target_col_idx,
                        train_df, test_df, idx_mapping, inverse_idx_mapping, idx_name_mapping):
    # Column info (types, ranges/categories)
    col_info = {}
    for col_idx in num_col_idx:
        col_name = column_names[col_idx]
        col_info[col_idx] = {
            "type": "numerical",
            "max": float(pd.to_numeric(train_df[col_name]).max()),
            "min": float(pd.to_numeric(train_df[col_name]).min()),
        }
    for col_idx in cat_col_idx:
        col_name = column_names[col_idx]
        col_info[col_idx] = {
            "type": "categorical",
            "categorizes": sorted(list(set(train_df[col_name]))),
        }
    for col_idx in target_col_idx:
        col_name = column_names[col_idx]
        col_info[col_idx] = {
            "type": "categorical" if task_type == "binclass" else "numerical",
        }
        if task_type == "binclass":
            col_info[col_idx]["categorizes"] = sorted(list(set(train_df[col_name])))
        else:
            col_info[col_idx]["max"] = float(pd.to_numeric(train_df[col_name]).max())
            col_info[col_idx]["min"] = float(pd.to_numeric(train_df[col_name]).min())

    # SDV-style metadata
    metadata = {"columns": {}}
    for i in num_col_idx:
        metadata["columns"][i] = {"sdtype": "numerical", "computer_representation": "Float"}
    for i in cat_col_idx:
        metadata["columns"][i] = {"sdtype": "categorical"}
    for i in target_col_idx:
        metadata["columns"][i] = {"sdtype": "categorical" if task_type == "binclass" else "numerical"}
        if task_type != "binclass":
            metadata["columns"][i]["computer_representation"] = "Float"

    info = {
        "name": name,
        "task_type": task_type,
        "header": "infer",
        "column_names": column_names,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "file_type": "csv",
        "data_path": str(TRAIN_CSV.as_posix()),
        "test_path": str(TEST_CSV.as_posix()),
        "column_info": col_info,
        "train_num": int(train_df.shape[0]),
        "test_num": int(test_df.shape[0]),
        "idx_mapping": idx_mapping,
        "inverse_idx_mapping": inverse_idx_mapping,
        "idx_name_mapping": idx_name_mapping,
        "metadata": metadata,
    }
    return info


def main():
    print(f"Downloading Kaggle dataset: {DATASET_SLUG}")
    download_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    print(f"Downloaded to: {download_path}")

    src_csv = find_source_csv(download_path)
    print(f"Using source CSV: {src_csv}")

    df = read_csv_robust(src_csv)

    # Remove id column
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Verify required columns exist
    for c in NUM_COLUMNS + CAT_COLUMNS + [TARGET_COLUMN]:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found. Found columns: {list(df.columns)}")

    # Cast numeric columns to float32
    for c in NUM_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    # Ensure categorical columns are ints (TabDDPM expects numeric categorical encoding)
    for c in CAT_COLUMNS + [TARGET_COLUMN]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.int64)

    # Replace '?' placeholders if any
    for col in NUM_COLUMNS:
        df.loc[df[col].astype(str) == "?", col] = np.nan
    for col in CAT_COLUMNS:
        df.loc[df[col].astype(str) == "?", col] = -1  # mark as a separate category if needed

    # Compute indices based on current column order
    column_names = list(df.columns)
    num_col_idx = [column_names.index(c) for c in NUM_COLUMNS]
    cat_col_idx = [column_names.index(c) for c in CAT_COLUMNS]
    target_col_idx = [column_names.index(TARGET_COLUMN)]

    # 90/10 split ensuring categorical coverage
    train_df, test_df, used_seed = ensure_categorical_coverage_split(
        df, CAT_COLUMNS + [TARGET_COLUMN], test_ratio=TEST_RATIO, seed_start=RANDOM_STATE
    )
    print(f"Split seed used: {used_seed}")
    print(f"{DATASET_NAME} train/test/total: {train_df.shape} / {test_df.shape} / {df.shape}")

    # Save npy arrays
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INFO_DIR.mkdir(parents=True, exist_ok=True)
    SYN_DIR.mkdir(parents=True, exist_ok=True)

    X_num_train = train_df[NUM_COLUMNS].to_numpy(dtype=np.float32)
    X_cat_train = train_df[CAT_COLUMNS].to_numpy(dtype=np.int64)
    y_train = train_df[[TARGET_COLUMN]].to_numpy(dtype=np.int64)

    X_num_test = test_df[NUM_COLUMNS].to_numpy(dtype=np.float32)
    X_cat_test = test_df[CAT_COLUMNS].to_numpy(dtype=np.int64)
    y_test = test_df[[TARGET_COLUMN]].to_numpy(dtype=np.int64)

    np.save(DATA_DIR / "X_num_train.npy", X_num_train)
    np.save(DATA_DIR / "X_cat_train.npy", X_cat_train)
    np.save(DATA_DIR / "y_train.npy", y_train)

    np.save(DATA_DIR / "X_num_test.npy", X_num_test)
    np.save(DATA_DIR / "X_cat_test.npy", X_cat_test)
    np.save(DATA_DIR / "y_test.npy", y_test)

    print("Saved .npy files for train/test.")

    # Save train/test CSVs (numeric as float32)
    train_df[NUM_COLUMNS] = train_df[NUM_COLUMNS].astype(np.float32)
    test_df[NUM_COLUMNS] = test_df[NUM_COLUMNS].astype(np.float32)
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    # Also save synthetic copies expected by evaluation scripts
    train_df.to_csv(SYN_REAL_CSV, index=False)
    test_df.to_csv(SYN_TEST_CSV, index=False)

    # Build mappings and info.json
    idx_mapping, inverse_idx_mapping, idx_name_mapping = build_mappings(
        column_names, num_col_idx, cat_col_idx, target_col_idx
    )
    info = build_metadata_info(
        DATASET_NAME, "binclass", column_names, num_col_idx, cat_col_idx, target_col_idx,
        train_df, test_df, idx_mapping, inverse_idx_mapping, idx_name_mapping
    )

    with open(INFO_JSON_CARDIO, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
    with open(INFO_JSON_GLOBAL, "w", encoding="utf-8") as f:
        json.dump({
            "name": DATASET_NAME,
            "task_type": "binclass",
            "header": "infer",
            "column_names": None,
            "num_col_idx": num_col_idx,
            "cat_col_idx": cat_col_idx,
            "target_col_idx": target_col_idx,
            "file_type": "csv",
            "data_path": str(TRAIN_CSV.as_posix()),
            "test_path": str(TEST_CSV.as_posix()),
        }, f, indent=4)

    print("Wrote info.json and data/Info/cardio.json.")

if __name__ == "__main__":
    main()