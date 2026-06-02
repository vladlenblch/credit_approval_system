import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COL = "SeriousDlqin2yrs"
DEFAULT_DECISION_THRESHOLD = 0.5

RAW_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

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


class WeightedBlendModel:
    def __init__(self):
        self.impute_values = self._load_impute_values()
        self.feature_columns = self._load_feature_columns()
        self.weights = self._load_weights()
        self.decision_threshold = self._load_decision_threshold()
        self.models = self._load_models()

    def predict(self, payload):
        features = self._preprocess(payload)

        model_scores = {
            "catboost": self._predict_catboost(features),
            "lightgbm": self._predict_joblib_models("lightgbm", features),
            "xgboost": self._predict_joblib_models("xgboost", features),
        }

        default_probability = sum(
            self.weights[model_name] * score
            for model_name, score in model_scores.items()
        )

        return {
            "default_probability": default_probability,
            "approval_decision": "reject" if default_probability >= self.decision_threshold else "approve",
            "decision_threshold": self.decision_threshold,
            "model_scores": model_scores,
            "blend_weights": self.weights,
        }

    def _load_impute_values(self):
        metadata_path = PROCESSED_DATA_DIR / "prepare_data_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata["numeric_impute_values"]

    def _load_feature_columns(self):
        train_path = PROCESSED_DATA_DIR / "train.csv"
        columns = pd.read_csv(train_path, nrows=0).columns.tolist()
        return [column for column in columns if column != TARGET_COL]

    def _load_weights(self):
        metrics_path = ARTIFACTS_DIR / "ensemble" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return metrics["weighted_blend"]["weights"]

    def _load_decision_threshold(self):
        threshold_path = ARTIFACTS_DIR / "ensemble" / "threshold_metrics.json"

        if not threshold_path.exists():
            return DEFAULT_DECISION_THRESHOLD

        metrics = json.loads(threshold_path.read_text(encoding="utf-8"))
        return metrics["best_threshold"]

    def _load_models(self):
        return {
            "catboost": self._load_catboost_models(),
            "lightgbm": self._load_joblib_models("lightgbm"),
            "xgboost": self._load_joblib_models("xgboost"),
        }

    def _load_catboost_models(self):
        models = []
        model_paths = sorted((ARTIFACTS_DIR / "catboost" / "fold_models").glob("*.cbm"))

        for model_path in model_paths:
            model = CatBoostClassifier()
            model.load_model(model_path)
            models.append(model)

        if not models:
            raise FileNotFoundError("CatBoost fold models were not found.")

        return models

    def _load_joblib_models(self, model_name):
        model_paths = sorted((ARTIFACTS_DIR / model_name / "fold_models").glob("*.joblib"))
        models = [joblib.load(model_path) for model_path in model_paths]

        if not models:
            raise FileNotFoundError(f"{model_name} fold models were not found.")

        return models

    def _preprocess(self, payload):
        row = {}

        for feature in RAW_FEATURES:
            value = payload.get(feature)
            row[feature] = np.nan if value in ("", None) else float(value)

        df = pd.DataFrame([row])

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

        for col in NUMERIC_IMPUTE_COLS:
            df[col] = df[col].fillna(self.impute_values[col])

        return df[self.feature_columns]

    def _predict_catboost(self, features):
        predictions = [
            model.predict_proba(features)[:, 1][0]
            for model in self.models["catboost"]
        ]
        return float(np.mean(predictions))

    def _predict_joblib_models(self, model_name, features):
        predictions = [
            model.predict_proba(features)[:, 1][0]
            for model in self.models[model_name]
        ]
        return float(np.mean(predictions))


@lru_cache(maxsize=1)
def get_model_service():
    return WeightedBlendModel()
