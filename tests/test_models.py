"""Unit and integration tests for HexaMind RUL models and training pipeline."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.rul import DLTrainConfig, SequenceRULTrainer, XGBoostRULModel  # noqa: E402


def _make_synthetic_windows(n_samples: int = 80, window: int = 30, n_features: int = 62, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.5, 0.2, size=(n_samples, window, n_features)).astype(np.float32)
    # Smooth synthetic RUL target correlated with feature mean
    y = np.clip(125.0 - X.mean(axis=(1, 2)) * 80.0 + rng.normal(0, 5.0, size=n_samples), 0.0, 125.0).astype(np.float32)
    return X, y


def test_xgboost_fit_predict_uncertainty_and_save_load(tmp_path: Path):
    X_train, y_train = _make_synthetic_windows(n_samples=120, seed=10)
    X_val, y_val = _make_synthetic_windows(n_samples=40, seed=20)
    X_train_2d = X_train[:, -1, :]
    X_val_2d = X_val[:, -1, :]

    model = XGBoostRULModel(n_estimators=25, max_depth=3, learning_rate=0.1, random_state=42)
    assert model.lower_model.get_params()["quantile_alpha"] == pytest.approx(0.05)
    assert model.upper_model.get_params()["quantile_alpha"] == pytest.approx(0.95)
    model.fit(X_train_2d, y_train, X_val_2d, y_val)

    preds = model.predict(X_val_2d)
    assert preds.shape == (40,)
    assert np.isfinite(preds).all()

    unc = model.predict_with_uncertainty(X_val_2d)
    assert set(unc.keys()) >= {"rul_mean", "rul_lower", "rul_upper"}
    assert unc["rul_mean"].shape == (40,)
    assert (unc["rul_lower"] <= unc["rul_upper"]).all()

    save_path = tmp_path / "xgb_test.json"
    model.save(str(save_path))
    assert save_path.is_file()

    loaded = XGBoostRULModel.load(str(save_path))
    loaded_preds = loaded.predict(X_val_2d)
    np.testing.assert_allclose(preds, loaded_preds, rtol=1e-5, atol=1e-5)


def test_xgboost_import_does_not_require_torch():
    code = (
        "import sys; "
        f"sys.path.insert(0, r'{SRC_DIR}'); "
        "from models.rul import XGBoostRULModel; "
        "assert 'torch' not in sys.modules, 'torch was eagerly imported'"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize("arch", ["lstm", "gru", "cnn1d"])
def test_sequence_models_fit_uncertainty_and_save_load(arch: str, tmp_path: Path):
    X_train, y_train = _make_synthetic_windows(n_samples=96, seed=100)
    X_val, y_val = _make_synthetic_windows(n_samples=32, seed=200)

    cfg = DLTrainConfig(
        arch=arch,
        input_dim=X_train.shape[-1],
        hidden_dim=16,
        num_layers=1,
        dropout=0.2,
        lr=2e-3,
        batch_size=32,
        epochs=2,
        patience=2,
        mc_samples=8,
        seed=42,
    )
    trainer = SequenceRULTrainer(cfg)
    trainer.fit(X_train, y_train, X_val, y_val)

    unc = trainer.predict_with_uncertainty(X_val)
    assert set(unc.keys()) >= {"rul_mean", "rul_lower", "rul_upper"}
    assert unc["rul_mean"].shape == (32,)
    assert np.isfinite(unc["rul_mean"]).all()
    assert (unc["rul_lower"] <= unc["rul_upper"]).all()

    ckpt_path = tmp_path / f"{arch}_test.pt"
    trainer.save(str(ckpt_path))
    assert ckpt_path.is_file()

    loaded = SequenceRULTrainer.load(str(ckpt_path))
    loaded_unc = loaded.predict_with_uncertainty(X_val)
    assert loaded_unc["rul_mean"].shape == (32,)
    assert np.isfinite(loaded_unc["rul_mean"]).all()


def test_training_summary_schema_if_present():
    summary_path = PROJECT_ROOT / "models" / "training_summary.json"
    if not summary_path.is_file():
        pytest.skip("models/training_summary.json not present yet")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "best_model_by_val_rmse" in data
    assert "runs" in data and len(data["runs"]) >= 1
    for run in data["runs"]:
        assert "artifact" in run
        assert "\\" not in run["artifact"]
