import numpy as np
import pandas as pd

from model_utils import ARTIFACTS_DIR, save_json


ENSEMBLE_DIR = ARTIFACTS_DIR / "ensemble"
PREDICTION_COL = "weighted_blend_pred"

PROFIT_GOOD_APPROVED = 1.0
LOSS_BAD_APPROVED = 5.0
N_THRESHOLDS = 1001


def calculate_business_metrics(y_true, y_pred_proba, threshold):
    approved = y_pred_proba < threshold
    predicted_default = ~approved

    good_approved = int(((y_true == 0) & approved).sum())
    bad_approved = int(((y_true == 1) & approved).sum())
    good_rejected = int(((y_true == 0) & predicted_default).sum())
    bad_rejected = int(((y_true == 1) & predicted_default).sum())

    profit = (
        good_approved * PROFIT_GOOD_APPROVED
        - bad_approved * LOSS_BAD_APPROVED
    )

    precision = (
        bad_rejected / (bad_rejected + good_rejected)
        if bad_rejected + good_rejected > 0
        else 0
    )
    recall = (
        bad_rejected / (bad_rejected + bad_approved)
        if bad_rejected + bad_approved > 0
        else 0
    )

    return {
        "threshold": float(threshold),
        "profit": float(profit),
        "approval_rate": float(approved.mean()),
        "rejection_rate": float(predicted_default.mean()),
        "approved_count": int(approved.sum()),
        "rejected_count": int(predicted_default.sum()),
        "good_approved": good_approved,
        "bad_approved": bad_approved,
        "good_rejected": good_rejected,
        "bad_rejected": bad_rejected,
        "precision_rejected": float(precision),
        "recall_bad_rejected": float(recall),
    }


def find_best_threshold(y_true, y_pred_proba):
    thresholds = np.linspace(0.0, 1.0, N_THRESHOLDS)
    metrics = [
        calculate_business_metrics(y_true, y_pred_proba, threshold)
        for threshold in thresholds
    ]

    return max(metrics, key=lambda item: item["profit"])


def load_predictions(dataset_name):
    return pd.read_csv(ENSEMBLE_DIR / f"{dataset_name}_predictions.csv")


def main():
    oof_df = load_predictions("oof")
    test_df = load_predictions("test")

    best_oof_metrics = find_best_threshold(
        oof_df["target"],
        oof_df[PREDICTION_COL],
    )
    best_threshold = best_oof_metrics["threshold"]
    test_metrics = calculate_business_metrics(
        test_df["target"],
        test_df[PREDICTION_COL],
        best_threshold,
    )
    default_threshold_oof_metrics = calculate_business_metrics(
        oof_df["target"],
        oof_df[PREDICTION_COL],
        0.5,
    )
    default_threshold_test_metrics = calculate_business_metrics(
        test_df["target"],
        test_df[PREDICTION_COL],
        0.5,
    )

    result = {
        "prediction_col": PREDICTION_COL,
        "profit_good_approved": PROFIT_GOOD_APPROVED,
        "loss_bad_approved": LOSS_BAD_APPROVED,
        "best_threshold": best_threshold,
        "oof": {
            "best_threshold": best_oof_metrics,
            "default_threshold_0_5": default_threshold_oof_metrics,
        },
        "test": {
            "best_threshold": test_metrics,
            "default_threshold_0_5": default_threshold_test_metrics,
        },
    }

    save_json(result, ENSEMBLE_DIR / "threshold_metrics.json")

    print("Best threshold: ", best_threshold)
    print("OOF best threshold metrics: ", best_oof_metrics)
    print("OOF default threshold metrics: ", default_threshold_oof_metrics)
    print("Test best threshold metrics: ", test_metrics)
    print("Test default threshold metrics: ", default_threshold_test_metrics)
    print(f"Saved threshold metrics to {ENSEMBLE_DIR / 'threshold_metrics.json'}")


if __name__ == "__main__":
    main()
