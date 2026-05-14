import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
VALID_SIZE = 0.15
TEST_SIZE = 0.15

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "needed_datasets"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COL = "SeriousDlqin2yrs"
ID_COL = "Unnamed: 0"

LATE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

NUMERIC_IMPUTE_COLS = [
    "age",
    "MonthlyIncome",
    "NumberOfDependents",
]

def split_datasets(df, valid_size, test_size, random_state):
    if valid_size <= 0 or test_size <= 0:
        raise ValueError("valid_size and test_size must be positive.")
    if valid_size + test_size >= 1:
        raise ValueError("valid_size + test_size must be less than 1.")

    temp_size = valid_size + test_size
    valid_share_from_temp = valid_size / temp_size

    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=random_state,
        stratify=df[TARGET_COL],
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=1 - valid_share_from_temp,
        random_state=random_state,
        stratify=temp_df[TARGET_COL],
    )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

def add_features(df):
    df = df.copy()

    df["MonthlyIncome_missing"] = df["MonthlyIncome"].isna().astype(int)
    df["NumberOfDependents_missing"] = df["NumberOfDependents"].isna().astype(int)
    df["age_eq_0"] = (df["age"] == 0).astype(int)
    df["has_special_past_due_code"] = df[LATE_COLS].ge(90).any(axis=1).astype(int)
    df["RevolvingUtilizationOfUnsecuredLines_gt_1"] = (
        df["RevolvingUtilizationOfUnsecuredLines"] > 1
    ).astype(int)

    df.loc[df["age"] == 0, "age"] = np.nan

    df["total_past_due"] = df[LATE_COLS].sum(axis=1)
    df["has_past_due"] = (df["total_past_due"] > 0).astype(int)

    return df

def fit_impute_values(train_df):
    return {
        col: float(train_df[col].median())
        for col in NUMERIC_IMPUTE_COLS
    }

def apply_preprocessing(df, impute_values):
    df = add_features(df)

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    for col, value in impute_values.items():
        df[col] = df[col].fillna(value)

    return df

def build_datasets():
    train_raw_path = INPUT_DIR / "cs-training.csv"
    train_raw = pd.read_csv(train_raw_path)

    train_raw, valid_raw, test_raw = split_datasets(
        train_raw,
        valid_size=VALID_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_with_features = add_features(train_raw)
    impute_values = fit_impute_values(train_with_features)

    train_processed = apply_preprocessing(train_raw, impute_values)
    valid_processed = apply_preprocessing(valid_raw, impute_values)
    test_processed = apply_preprocessing(test_raw, impute_values)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_processed.to_csv(OUTPUT_DIR / "train.csv", index=False)
    valid_processed.to_csv(OUTPUT_DIR / "valid.csv", index=False)
    test_processed.to_csv(OUTPUT_DIR / "test.csv", index=False)

    metadata = {
        "random_state": RANDOM_STATE,
        "valid_size": VALID_SIZE,
        "test_size": TEST_SIZE,
        "target_col": TARGET_COL,
        "dropped_columns": [ID_COL],
        "late_cols": LATE_COLS,
        "numeric_impute_values": impute_values,
        "rows": {
            "train": len(train_processed),
            "valid": len(valid_processed),
            "test": len(test_processed),
        },
        "target_rate": {
            "train": float(train_processed[TARGET_COL].mean()),
            "valid": float(valid_processed[TARGET_COL].mean()),
            "test": float(test_processed[TARGET_COL].mean()),
        },
    }

    (OUTPUT_DIR / "prepare_data_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved processed datasets to {OUTPUT_DIR}")
    print(
        pd.DataFrame(
            {
                "dataset": ["train", "valid", "test"],
                "rows": [
                    len(train_processed),
                    len(valid_processed),
                    len(test_processed),
                ],
                "columns": [
                    train_processed.shape[1],
                    valid_processed.shape[1],
                    test_processed.shape[1],
                ],
                "target_rate": [
                    train_processed[TARGET_COL].mean(),
                    valid_processed[TARGET_COL].mean(),
                    test_processed[TARGET_COL].mean(),
                ],
            }
        ).round(4)
    )

def main():
    build_datasets()

if __name__ == "__main__":
    main()
