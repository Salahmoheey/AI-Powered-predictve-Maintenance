import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import validate_data, remove_constant_sensors
from features import create_features, create_windows, build_online_features
from validation import get_group_kfold_splits, save_group_kfold_unit_ids


def sample_frame(n_units=2, cycles=35):
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, cycles + 1):
            row = {"unit_id": unit, "cycle": cycle, "setting_1": 0.1,
                   "setting_2": 0.2, "setting_3": 0.3}
            row.update({f"sensor_{i}": float(cycle + i) for i in range(1, 22)})
            rows.append(row)
    return pd.DataFrame(rows, columns=["unit_id", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)])


def test_validation_rejects_duplicate_cycle():
    frame = sample_frame()
    frame.loc[1, "cycle"] = 1
    with pytest.raises(ValueError, match="duplicate"):
        validate_data(frame, "sample")


def test_constant_columns_are_detected_from_train_only():
    train = sample_frame()
    validation = sample_frame()
    test = sample_frame()
    for frame in (train, validation, test):
        frame["sensor_1"] = 4.0
    result = remove_constant_sensors(train, validation, test)
    assert "sensor_1" in result[3]
    assert "sensor_1" not in result[0].columns


def test_constant_setting_3_is_dropped_but_varying_setting_is_kept():
    train = sample_frame()
    train["setting_3"] = 100.0
    train["setting_1"] = np.linspace(0.0, 1.0, len(train))
    result = remove_constant_sensors(train, train.copy(), train.copy())
    assert "setting_3" in result[3]
    assert "setting_3" not in result[0].columns
    assert "setting_1" in result[0].columns


def test_features_use_30_cycle_window():
    frame = sample_frame(n_units=1, cycles=35)
    featured = create_features(frame, window=30)
    assert len(featured) == 35
    assert "sensor_1_rolling_mean" in featured
    assert np.isclose(featured.loc[29, "sensor_1_rolling_mean"], frame.loc[:29, "sensor_1"].mean())


def test_window_builder_returns_30_cycle_sequences():
    frame = sample_frame(n_units=1, cycles=31)
    frame["RUL"] = np.arange(31, 0, -1)
    metadata, sequences, labels = create_windows(frame, window=30)
    assert sequences.shape == (2, 30, 24)
    assert len(metadata) == len(labels) == 2


def test_group_kfold_keeps_engines_separate():
    frame = sample_frame(n_units=10, cycles=3)
    for train_idx, valid_idx in get_group_kfold_splits(frame, n_splits=5):
        train_units = set(frame.iloc[train_idx].unit_id)
        valid_units = set(frame.iloc[valid_idx].unit_id)
        assert train_units.isdisjoint(valid_units)


def test_group_kfold_saves_engine_ids(tmp_path):
    frame = sample_frame(n_units=10, cycles=3)
    assert save_group_kfold_unit_ids(frame, tmp_path, n_splits=5) == 5
    assert (tmp_path / "fold_1_train_units.csv").exists()
    assert (tmp_path / "fold_1_validation_units.csv").exists()


def test_online_feature_builder_scales_and_generates_features():
    from sklearn.preprocessing import MinMaxScaler
    kept = ["setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
    scaler = MinMaxScaler().fit(pd.DataFrame([[0.0] * len(kept), [100.0] * len(kept)], columns=kept))
    artifacts = {"scaler": scaler, "dropped_columns": [], "remaining_scale_columns": kept,
                 "feature_columns": ["unit_id", "cycle"] + kept}
    reading = {"unit_id": 1, "cycle": 1, **{column: 10.0 for column in kept}}
    features, history = build_online_features(reading, pd.DataFrame(), artifacts)
    assert "sensor_1_rolling_mean" in features
    assert len(history) == 1
    assert features["sensor_1"] == pytest.approx(0.1)
