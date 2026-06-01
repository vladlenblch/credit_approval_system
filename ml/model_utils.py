from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TRAIN_DATASET = "train.csv"
TEST_DATASET = "test.csv"
VALID_DATASET = "valid.csv"
TARGET = "SeriousDlqin2yrs"


def load_datasets():
    train_df = pd.read_csv(DATASETS_DIR / TRAIN_DATASET)
    valid_df = pd.read_csv(DATASETS_DIR / VALID_DATASET)
    test_df = pd.read_csv(DATASETS_DIR / TEST_DATASET)

    return train_df, valid_df, test_df


def split_xy(dataset):
    X = dataset.drop(columns=[TARGET])
    y = dataset[TARGET]

    return X, y


def evaluate_predictions(y_true, y_pred_proba):
    return {
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
    }


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )
