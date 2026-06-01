import optuna
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from model_utils import (
    ARTIFACTS_DIR,
    evaluate_predictions,
    load_datasets,
    save_json,
    split_xy,
)


RANDOM_STATE = 42
N_TRIALS = 50
N_SPLITS = 3
EARLY_STOPPING_ROUNDS = 100
MODEL_DIR = ARTIFACTS_DIR / "catboost"


def suggest_params(trial):
    params = {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": RANDOM_STATE,
        "verbose": 100,
        "allow_writing_files": False,
        "iterations": trial.suggest_int("iterations", 800, 5000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.3, 100.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.01, 20.0, log=True),
        "border_count": trial.suggest_int("border_count", 64, 255),
        "rsm": trial.suggest_float("rsm", 0.6, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 80),
        "leaf_estimation_iterations": trial.suggest_int(
            "leaf_estimation_iterations",
            1,
            8,
        ),
        "auto_class_weights": trial.suggest_categorical(
            "auto_class_weights",
            [None, "Balanced", "SqrtBalanced"],
        ),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type",
            ["Bayesian", "Bernoulli", "MVS"],
        ),
    }

    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature",
            0.0,
            10.0,
        )
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

    return params


def objective(trial, X, y):
    params = suggest_params(trial)
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_metrics = []
    best_iterations = []

    for train_idx, valid_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        model = CatBoostClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            use_best_model=True,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )

        valid_pred = model.predict_proba(X_valid)[:, 1]
        fold_metrics.append(evaluate_predictions(y_valid, valid_pred))
        best_iterations.append(model.get_best_iteration())

    mean_pr_auc = sum(metric["pr_auc"] for metric in fold_metrics) / N_SPLITS
    mean_roc_auc = sum(metric["roc_auc"] for metric in fold_metrics) / N_SPLITS
    mean_best_iteration = round(sum(best_iterations) / N_SPLITS)

    trial.set_user_attr("fold_metrics", fold_metrics)
    trial.set_user_attr("roc_auc", mean_roc_auc)
    trial.set_user_attr("best_iteration", mean_best_iteration)

    return mean_pr_auc


def train_final_model(best_params, X_train, y_train):
    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train)

    return model


def train_oof_models(best_params, X, y, X_test):
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_pred = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_metrics = []
    fold_models_dir = MODEL_DIR / "fold_models"
    fold_models_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        model = CatBoostClassifier(**best_params)
        model.fit(X_train, y_train)

        valid_pred = model.predict_proba(X_valid)[:, 1]
        oof_pred[valid_idx] = valid_pred
        test_pred += model.predict_proba(X_test)[:, 1] / N_SPLITS

        fold_metrics.append(evaluate_predictions(y_valid, valid_pred))
        model.save_model(fold_models_dir / f"model_fold_{fold}.cbm")

    oof_metrics = evaluate_predictions(y, oof_pred)

    return oof_pred, test_pred, oof_metrics, fold_metrics


def main():
    train_df, valid_df, test_df = load_datasets()
    dev_df = pd.concat([train_df, valid_df], ignore_index=True)

    X_dev, y_dev = split_xy(dev_df)
    X_test, y_test = split_xy(test_df)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_dev, y_dev),
        n_trials=N_TRIALS,
    )

    print("Best CV PR-AUC: ", study.best_value)
    print("Best params: ", study.best_params)

    best_params = {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": RANDOM_STATE,
        "verbose": 100,
        "allow_writing_files": False,
        **study.best_params,
    }
    best_params["iterations"] = study.best_trial.user_attrs["best_iteration"]

    oof_pred, cv_test_pred, oof_metrics, oof_fold_metrics = train_oof_models(
        best_params,
        X_dev,
        y_dev,
        X_test,
    )

    model = train_final_model(best_params, X_dev, y_dev)

    test_pred = model.predict_proba(X_test)[:, 1]

    test_metrics = evaluate_predictions(y_test, test_pred)
    cv_test_metrics = evaluate_predictions(y_test, cv_test_pred)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_DIR / "model.cbm")
    pd.DataFrame(
        {
            "target": y_dev,
            "catboost_pred": oof_pred,
        }
    ).to_csv(MODEL_DIR / "oof_predictions.csv", index=False)
    pd.DataFrame(
        {
            "target": y_test,
            "catboost_pred": cv_test_pred,
        }
    ).to_csv(MODEL_DIR / "test_predictions.csv", index=False)
    save_json(best_params, MODEL_DIR / "best_params.json")
    save_json(
        {
            "cv": {
                "roc_auc": study.best_trial.user_attrs.get("roc_auc"),
                "pr_auc": study.best_value,
                "fold_metrics": study.best_trial.user_attrs.get("fold_metrics"),
            },
            "oof": {
                "metrics": oof_metrics,
                "fold_metrics": oof_fold_metrics,
            },
            "test": {
                "single_model": test_metrics,
                "cv_ensemble": cv_test_metrics,
            },
            "best_trial": {
                "number": study.best_trial.number,
                "pr_auc": study.best_value,
                "roc_auc": study.best_trial.user_attrs.get("roc_auc"),
                "best_iteration": study.best_trial.user_attrs.get("best_iteration"),
            },
            "n_trials": N_TRIALS,
            "n_splits": N_SPLITS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        },
        MODEL_DIR / "metrics.json",
    )

    print("CV PR-AUC: ", study.best_value)
    print("OOF metrics: ", oof_metrics)
    print("Test metrics single model: ", test_metrics)
    print("Test metrics CV ensemble: ", cv_test_metrics)
    print(f"Saved CatBoost artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
