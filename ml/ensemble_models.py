import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression

from model_utils import ARTIFACTS_DIR, evaluate_predictions, save_json


RANDOM_STATE = 42
N_TRIALS = 300
MODEL_NAMES = ["catboost", "lightgbm", "xgboost"]
ENSEMBLE_DIR = ARTIFACTS_DIR / "ensemble"


def load_predictions(dataset_name):
    result = None

    for model_name in MODEL_NAMES:
        path = ARTIFACTS_DIR / model_name / f"{dataset_name}_predictions.csv"
        pred_col = f"{model_name}_pred"
        model_df = pd.read_csv(path)

        if result is None:
            result = model_df[["target", pred_col]].rename(
                columns={pred_col: model_name}
            )
        else:
            if not result["target"].equals(model_df["target"]):
                raise ValueError(f"Target mismatch in {path}")
            result[model_name] = model_df[pred_col]

    return result


def normalize_weights(weights):
    weights = np.array(weights, dtype=float)
    weights_sum = weights.sum()

    if weights_sum == 0:
        return np.ones(len(weights)) / len(weights)

    return weights / weights_sum


def weighted_prediction(X, weights):
    weights = normalize_weights(weights)
    return X.to_numpy().dot(weights)


def tune_blend_weights(X_oof, y_oof):
    def objective(trial):
        weights = [
            trial.suggest_float(model_name, 0.0, 1.0)
            for model_name in MODEL_NAMES
        ]
        pred = weighted_prediction(X_oof, weights)
        return evaluate_predictions(y_oof, pred)["pr_auc"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    weights = normalize_weights([
        study.best_params[model_name]
        for model_name in MODEL_NAMES
    ])

    return study, weights


def fit_stacking_model(X_oof, y_oof):
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_oof, y_oof)

    return model


def collect_metrics(y_oof, y_test, oof_pred, test_pred):
    return {
        "oof": evaluate_predictions(y_oof, oof_pred),
        "test": evaluate_predictions(y_test, test_pred),
    }


def main():
    oof_df = load_predictions("oof")
    test_df = load_predictions("test")

    X_oof = oof_df[MODEL_NAMES]
    y_oof = oof_df["target"]
    X_test = test_df[MODEL_NAMES]
    y_test = test_df["target"]

    individual_metrics = {
        model_name: collect_metrics(
            y_oof,
            y_test,
            X_oof[model_name],
            X_test[model_name],
        )
        for model_name in MODEL_NAMES
    }

    simple_oof_pred = X_oof.mean(axis=1)
    simple_test_pred = X_test.mean(axis=1)
    simple_metrics = collect_metrics(
        y_oof,
        y_test,
        simple_oof_pred,
        simple_test_pred,
    )

    blend_study, blend_weights = tune_blend_weights(X_oof, y_oof)
    weighted_oof_pred = weighted_prediction(X_oof, blend_weights)
    weighted_test_pred = weighted_prediction(X_test, blend_weights)
    weighted_metrics = collect_metrics(
        y_oof,
        y_test,
        weighted_oof_pred,
        weighted_test_pred,
    )

    stacking_model = fit_stacking_model(X_oof, y_oof)
    stacking_oof_pred = stacking_model.predict_proba(X_oof)[:, 1]
    stacking_test_pred = stacking_model.predict_proba(X_test)[:, 1]
    stacking_metrics = collect_metrics(
        y_oof,
        y_test,
        stacking_oof_pred,
        stacking_test_pred,
    )

    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(stacking_model, ENSEMBLE_DIR / "logreg_stacking.joblib")

    pd.DataFrame(
        {
            "target": y_oof,
            "simple_average_pred": simple_oof_pred,
            "weighted_blend_pred": weighted_oof_pred,
            "logreg_stacking_pred": stacking_oof_pred,
        }
    ).to_csv(ENSEMBLE_DIR / "oof_predictions.csv", index=False)

    pd.DataFrame(
        {
            "target": y_test,
            "simple_average_pred": simple_test_pred,
            "weighted_blend_pred": weighted_test_pred,
            "logreg_stacking_pred": stacking_test_pred,
        }
    ).to_csv(ENSEMBLE_DIR / "test_predictions.csv", index=False)

    save_json(
        {
            "models": individual_metrics,
            "simple_average": simple_metrics,
            "weighted_blend": {
                "metrics": weighted_metrics,
                "weights": {
                    model_name: float(weight)
                    for model_name, weight in zip(MODEL_NAMES, blend_weights)
                },
                "best_trial": {
                    "number": blend_study.best_trial.number,
                    "pr_auc": float(blend_study.best_value),
                },
                "n_trials": N_TRIALS,
            },
            "logreg_stacking": {
                "metrics": stacking_metrics,
                "intercept": stacking_model.intercept_.tolist(),
                "coefficients": {
                    model_name: float(coef)
                    for model_name, coef in zip(
                        MODEL_NAMES,
                        stacking_model.coef_[0],
                    )
                },
            },
        },
        ENSEMBLE_DIR / "metrics.json",
    )

    print("Simple average: ", simple_metrics)
    print("Weighted blend: ", weighted_metrics)
    print("Weighted blend weights: ", dict(zip(MODEL_NAMES, blend_weights)))
    print("LogReg stacking: ", stacking_metrics)
    print(f"Saved ensemble artifacts to {ENSEMBLE_DIR}")


if __name__ == "__main__":
    main()
