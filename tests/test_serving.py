"""Tests for the raw-readings -> model-features recipe (src/serving.py)."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import WINDOW, create_features, create_windows, windowed_ewma  # noqa: E402
from preprocessing import COLUMNS, build_datasets  # noqa: E402
from serving import (  # noqa: E402
    ContractError,
    load_artifacts,
    raw_window_to_features,
    raw_window_to_sequence,
    readings_to_frame,
    required_channels,
    validate_readings,
)

GOLDEN = json.loads((ROOT / "artifacts" / "golden_example.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


@pytest.fixture(scope="module")
def pipeline():
    """The offline pipeline, run in memory (no files written)."""
    result = build_datasets()
    test = result["test"]
    subset = test[test["unit_id"].isin([1, 49])]
    return {"processed": subset, "features": create_features(subset)}


def make_readings(artifacts, cycles=2, start=None):
    start = start or datetime.now(timezone.utc) - timedelta(hours=1)
    readings = []
    for cycle in range(cycles):
        stamp = (start + timedelta(minutes=cycle)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for channel in required_channels(artifacts):
            readings.append({"sensor_type": channel, "value": 1.0 + cycle, "timestamp": stamp})
    return readings


def test_artifacts_describe_a_62_column_model_input(artifacts):
    order = artifacts["model_feature_columns"]
    assert len(order) == 62 and len(set(order)) == 62
    assert order == json.loads((ROOT / "artifacts" / "feature_order.json").read_text(encoding="utf-8"))
    assert not {"unit_id", "cycle", "RUL"} & set(order)
    assert "setting_3" in artifacts["dropped_columns"]
    assert len(required_channels(artifacts)) == 17


def test_golden_example_is_reproduced_by_the_recipe(artifacts):
    window = pd.DataFrame(GOLDEN["raw_window"])
    result = raw_window_to_features(window, artifacts, min_rows=30)
    assert list(result) == GOLDEN["feature_order"]
    for column, expected in GOLDEN["expected_features"].items():
        assert result[column] == pytest.approx(expected, abs=GOLDEN["tolerance"]), column


def test_recipe_matches_offline_features_for_every_cycle(artifacts, pipeline):
    """Serving (raw window -> vector) must equal training (features.py) for all cycles."""
    raw = pd.read_csv(ROOT / "data" / "raw" / "test_FD001.txt", sep=r"\s+", header=None, names=COLUMNS)
    offline = pipeline["features"]
    for unit in (1, 49):
        engine_raw = raw[raw["unit_id"] == unit].reset_index(drop=True)
        engine_off = offline[offline["unit_id"] == unit].reset_index(drop=True)
        step = 1 if len(engine_raw) < 100 else 7
        for index in range(0, len(engine_raw), step):
            window = engine_raw.iloc[max(0, index - 29) : index + 1]  # backend only keeps 30 rows
            result = raw_window_to_features(window, artifacts)
            expected = engine_off.loc[index, list(result)].to_numpy(dtype=float)
            assert np.allclose(list(result.values()), expected, atol=1e-9), (unit, index)


def test_raw_window_to_sequence_matches_offline_windows(artifacts, pipeline):
    """59 raw cycles -> (30, 62) sequence must match offline create_windows() exactly."""
    raw = pd.read_csv(ROOT / "data" / "raw" / "test_FD001.txt", sep=r"\s+", header=None, names=COLUMNS)
    offline = pipeline["features"]
    meta, X_off, _ = create_windows(offline, window=WINDOW, target_column="RUL", group_column="unit_id")

    for unit in (1, 49):
        engine_raw = raw[raw["unit_id"] == unit].reset_index(drop=True)
        unit_mask = (meta["unit_id"] == unit).to_numpy()
        unit_windows = X_off[unit_mask]
        # Check first, middle, and final valid sequence windows (requiring 59 raw rows)
        min_offset = WINDOW - 1; max_offset = len(unit_windows) - 1; sample_offsets = [] if max_offset < min_offset else [min_offset, (min_offset + max_offset) // 2, max_offset]
        for offset in sample_offsets:
            # Window `offset` ends at raw cycle index `WINDOW - 1 + offset` (0-based = 29 + offset)
            # and needs 59 preceding raw rows: indices [offset : offset + 59]
            start_idx = offset - (WINDOW - 1); raw_slice = engine_raw.iloc[start_idx : start_idx + (2 * WINDOW - 1)]
            seq = raw_window_to_sequence(raw_slice, artifacts, seq_len=WINDOW, window_size=WINDOW)
            assert seq.shape == (WINDOW, 62)
            assert seq.dtype == np.float32
            np.testing.assert_allclose(seq, unit_windows[offset], rtol=1e-5, atol=1e-6)


def test_raw_window_to_sequence_rejects_fewer_than_59_rows(artifacts):
    raw = pd.read_csv(ROOT / "data" / "raw" / "test_FD001.txt", sep=r"\s+", header=None, names=COLUMNS)
    short_window = raw[raw["unit_id"] == 1].head(58)
    with pytest.raises(ContractError) as error:
        raw_window_to_sequence(short_window, artifacts, seq_len=30, window_size=30)
    assert error.value.code == "INSUFFICIENT_DATA"


def test_windowed_ewma_matches_pandas_on_the_window():
    values = np.random.RandomState(1).rand(40)
    for size in (1, 2, 10, 30):
        chunk = values[-size:]
        assert windowed_ewma(chunk, 30) == pytest.approx(
            pd.Series(chunk).ewm(span=30, adjust=False).mean().iloc[-1], abs=1e-12
        )


def test_insufficient_data_error_code(artifacts):
    window = pd.DataFrame(GOLDEN["raw_window"]).head(5)
    with pytest.raises(ContractError) as error:
        raw_window_to_features(window, artifacts, min_rows=30)
    assert error.value.code == "INSUFFICIENT_DATA"


def test_valid_readings_are_pivoted_one_row_per_timestamp(artifacts):
    frame = readings_to_frame(make_readings(artifacts, cycles=3), artifacts)
    assert len(frame) == 3 and frame.index.is_monotonic_increasing
    assert set(required_channels(artifacts)) <= set(frame.columns)


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda r: r.pop(), "missing channels"),
        (lambda r: r[0].update(value=float("nan")), "finite"),
        (lambda r: r[0].update(value=float("inf")), "finite"),
        (lambda r: r[0].update(value="12"), "finite number"),
        (lambda r: r[0].update(sensor_type="sensor_99"), "unknown sensor_type"),
        (lambda r: r[0].update(timestamp="not-a-date"), "ISO-8601"),
        (lambda r: r[0].update(timestamp="2026-11-21T10:00:00"), "UTC offset"),
        (lambda r: r.append(dict(r[0])), "duplicate"),
    ],
)
def test_invalid_batches_are_rejected_whole(artifacts, mutate, fragment):
    readings = make_readings(artifacts, cycles=1)
    mutate(readings)
    with pytest.raises(ContractError, match=fragment) as error:
        validate_readings(readings, artifacts)
    assert error.value.code == "VALIDATION_ERROR"


def test_negative_sensor_reading_is_rejected_but_negative_setting_is_allowed(artifacts):
    readings = make_readings(artifacts, cycles=1)
    for row in readings:
        if row["sensor_type"] == "setting_1":
            row["value"] = -0.005
    assert len(validate_readings(readings, artifacts)) == 17

    for row in readings:
        if row["sensor_type"] == "sensor_2":
            row["value"] = -1.0
    with pytest.raises(ContractError, match="non-negative"):
        validate_readings(readings, artifacts)
