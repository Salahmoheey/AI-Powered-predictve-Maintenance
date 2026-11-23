"""Train and evaluate XGBoost, LSTM, GRU, and 1D-CNN RUL regressors on FD001."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import (  # noqa: E402
    WINDOW,
    create_features,
    create_windows,
    model_feature_columns,
)
from models.rul import DLTrainConfig, SequenceRULTrainer, XGBoostRULModel  # noqa: E402
from preprocessing import RUL_CAP  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODELS = ("xgboost", "lstm", "gru", "cnn1d")


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    cov = float(np.mean((y_true >= lower) & (y_true <= upper)))
    width = float(np.mean(upper - lower))
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "pi90_coverage": round(cov, 4),
        "pi90_mean_width": round(width, 2),
    }


def _last_cycle_indices(meta: pd.DataFrame) -> np.ndarray:
    return meta.groupby("unit_id", sort=False).tail(1).index.to_numpy()


def _build_splits() -> dict:
    splits = {}
    for name in ("train", "validation", "test"):
        path = PROCESSED_DIR / f"{name}_processed.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run `python src/preprocessing.py` first")
        processed = pd.read_csv(path)
        featured = create_features(processed, window=WINDOW)
        meta, X, y = create_windows(featured, window=WINDOW, target_column="RUL", group_column="unit_id")
        splits[name] = {
            "processed": processed,
            "featured": featured,
            "meta": meta,
            "X": X,
            "y": y.to_numpy(dtype=np.float32),
        }
    return splits


def _train_xgboost(splits: dict, seed: int) -> dict:
    # Use the last step of each 30-cycle window so XGBoost and DL models
    # are evaluated on the exact same cycles (from cycle 30 onward, 3,490 val rows).
    X_train = splits["train"]["X"][:, -1, :]
    y_train = splits["train"]["y"]
    groups = splits["train"]["meta"]["unit_id"].to_numpy()

    X_val = splits["validation"]["X"][:, -1, :]
    y_val = splits["validation"]["y"]

    X_test = splits["test"]["X"][:, -1, :]
    y_test = splits["test"]["y"]
    last_idx = _last_cycle_indices(splits["test"]["meta"])

    gkf = GroupKFold(n_splits=5)
    cv_rmses = []
    for fold_id, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        fold_model = XGBoostRULModel(random_state=seed + fold_id)
        fold_model.fit(X_train[tr_idx], y_train[tr_idx])
        fold_pred = fold_model.predict(X_train[va_idx])
        cv_rmses.append(float(np.sqrt(mean_squared_error(y_train[va_idx], fold_pred))))

    model = XGBoostRULModel(random_state=seed)
    model.fit(X_train, y_train, X_val, y_val)

    val_unc = model.predict_with_uncertainty(X_val)
    test_unc = model.predict_with_uncertainty(X_test)
    test_last_unc = model.predict_with_uncertainty(X_test[last_idx])

    artifact_path = MODELS_DIR / "xgboost_rul.joblib"
    model.save(str(artifact_path))

    return {
        "model": "xgboost",
        "hyperparameters": model.params,
        "cv_rmse_mean": round(float(np.mean(cv_rmses)), 4),
        "cv_rmse_std": round(float(np.std(cv_rmses)), 4),
        "validation": _metrics(y_val, val_unc["rul_mean"], val_unc["rul_lower"], val_unc["rul_upper"]),
        "test_all_cycles": _metrics(y_test, test_unc["rul_mean"], test_unc["rul_lower"], test_unc["rul_upper"]),
        "test_last_cycle": _metrics(
            y_test[last_idx], test_last_unc["rul_mean"], test_last_unc["rul_lower"], test_last_unc["rul_upper"]
        ),
        "artifact": artifact_path.relative_to(PROJECT_ROOT).as_posix(),
    }


def _train_dl_model(arch: str, splits: dict, input_dim: int, epochs: int, seed: int) -> dict:
    X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    X_val, y_val = splits["validation"]["X"], splits["validation"]["y"]
    X_test, y_test = splits["test"]["X"], splits["test"]["y"]
    last_idx = _last_cycle_indices(splits["test"]["meta"])

    cfg = DLTrainConfig(
        arch=arch,
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.25,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=64,
        epochs=epochs,
        patience=6,
        mc_samples=30,
        seed=seed,
    )
    trainer = SequenceRULTrainer(cfg)
    trainer.fit(X_train, y_train, X_val, y_val)

    val_unc = trainer.predict_with_uncertainty(X_val)
    test_unc = trainer.predict_with_uncertainty(X_test)
    test_last_unc = trainer.predict_with_uncertainty(X_test[last_idx])

    artifact_path = MODELS_DIR / f"{arch}_rul.pt"
    trainer.save(str(artifact_path))

    return {
        "model": arch,
        "hyperparameters": cfg.__dict__,
        "validation": _metrics(y_val, val_unc["rul_mean"], val_unc["rul_lower"], val_unc["rul_upper"]),
        "test_all_cycles": _metrics(y_test, test_unc["rul_mean"], test_unc["rul_lower"], test_unc["rul_upper"]),
        "test_last_cycle": _metrics(
            y_test[last_idx], test_last_unc["rul_mean"], test_last_unc["rul_lower"], test_last_unc["rul_upper"]
        ),
        "artifact": artifact_path.relative_to(PROJECT_ROOT).as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HexaMind RUL benchmark models.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    splits = _build_splits()
    feature_cols = model_feature_columns(splits["train"]["processed"].columns.tolist())
    input_dim = len(feature_cols)

    runs = []
    for name in args.models:
        print(f"=== Training {name.upper()} ===")
        if name == "xgboost":
            runs.append(_train_xgboost(splits, seed=args.seed))
        else:
            runs.append(_train_dl_model(name, splits, input_dim=input_dim, epochs=args.epochs, seed=args.seed))

    best_run = min(runs, key=lambda r: r["validation"]["rmse"])
    summary = {
        "best_model_by_val_rmse": best_run["model"],
        "window_size": WINDOW,
        "num_features": input_dim,
        "rul_cap": RUL_CAP,
        "runs": runs,
    }
    summary_path = MODELS_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {summary_path}")
    print(f"Best model by validation RMSE: {best_run['model']} ({best_run['validation']['rmse']})")


if __name__ == "__main__":
    main()
