"""Reference implementation of the "raw sensor readings -> model features" recipe.

The backend can import these functions directly or re-implement them following
docs/FEATURE_RECIPE.md; `artifacts/golden_example.json` lets it verify either choice.

Typical flow inside `POST /machines/{id}/predictions`:

    artifacts = load_artifacts()                                         # reused at start-up
    window = readings_to_frame(last_rows, artifacts)                     # at least 59 rows for DL
    seq = raw_window_to_sequence(window, artifacts, seq_len=30)          # shape (30, 62)
    model.predict_with_uncertainty(seq[None, ...])
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

try:  # imported as a package (e.g. src.serving)
    from .features import WINDOW, compute_window_features, get_sensor_columns
except ImportError:  # executed with src/ on sys.path
    from features import WINDOW, compute_window_features, get_sensor_columns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

ALLOWED_CHANNELS = [f"setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
MAX_BATCH_SIZE = 1000
FUTURE_TOLERANCE_SECONDS = 5


class ContractError(ValueError):
    """Input rejected by the API input contract (maps to an HTTP 4xx response)."""

    def __init__(self, code, message, http_status=422):
        super().__init__(message)
        self.code = code
        self.http_status = http_status


def load_artifacts(directory=ARTIFACT_DIR):
    """Load the fitted scaler and preprocessing metadata saved by preprocessing.py."""
    path = Path(directory) / "scaler.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `python src/preprocessing.py` first")
    artifacts = joblib.load(path)
    saved = artifacts.get("sklearn_version")
    if saved and saved != sklearn.__version__:
        warnings.warn(f"scaler saved with scikit-learn {saved}, running {sklearn.__version__}")
    return artifacts


def required_channels(artifacts):
    """Raw channels the API must receive for every cycle (the kept settings and sensors)."""
    return list(artifacts["remaining_scale_columns"])


def _parse_timestamp(value):
    if not isinstance(value, str):
        raise ContractError("VALIDATION_ERROR", "timestamp must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("VALIDATION_ERROR", f"timestamp {value!r} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ContractError("VALIDATION_ERROR", f"timestamp {value!r} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def validate_readings(readings, artifacts, now=None):
    """Validate one POST /sensor-logs body (`readings` list). The whole batch is rejected on error.

    Returns the readings as a list of (timestamp, sensor_type, value) tuples.
    """
    if not isinstance(readings, list) or not readings:
        raise ContractError("VALIDATION_ERROR", "readings must be a non-empty list")
    if len(readings) > MAX_BATCH_SIZE:
        raise ContractError("BATCH_TOO_LARGE", f"at most {MAX_BATCH_SIZE} readings per request", 413)
    now = now or datetime.now(timezone.utc)
    limit = now + timedelta(seconds=FUTURE_TOLERANCE_SECONDS)
    seen, parsed_rows = set(), []
    for index, item in enumerate(readings):
        if not isinstance(item, dict) or not {"sensor_type", "value", "timestamp"} <= set(item):
            raise ContractError("VALIDATION_ERROR", f"readings[{index}] needs sensor_type, value, timestamp")
        sensor_type = item["sensor_type"]
        if sensor_type not in ALLOWED_CHANNELS:
            raise ContractError("VALIDATION_ERROR", f"readings[{index}]: unknown sensor_type {sensor_type!r}")
        value = item["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ContractError("VALIDATION_ERROR", f"readings[{index}]: value must be a finite number")
        if sensor_type.startswith("sensor_") and value < 0:
            raise ContractError("VALIDATION_ERROR", f"readings[{index}]: sensor values must be non-negative")
        timestamp = _parse_timestamp(item["timestamp"])
        if timestamp > limit:
            raise ContractError("VALIDATION_ERROR", f"readings[{index}]: timestamp is in the future")
        key = (timestamp, sensor_type)
        if key in seen:
            raise ContractError("VALIDATION_ERROR", f"readings[{index}]: duplicate {sensor_type} at {item['timestamp']}")
        seen.add(key)
        parsed_rows.append((timestamp, sensor_type, float(value)))
    required = set(required_channels(artifacts))
    channels_by_time = {}
    for timestamp, sensor_type, _ in parsed_rows:
        channels_by_time.setdefault(timestamp, set()).add(sensor_type)
    for timestamp, channels in channels_by_time.items():
        missing = sorted(required - channels)
        if missing:
            raise ContractError("VALIDATION_ERROR", f"cycle at {timestamp.isoformat()} is missing channels: {missing}")
    return parsed_rows


def readings_to_frame(readings, artifacts, now=None):
    """Validate long-format readings and pivot them to one row per timestamp (oldest first)."""
    rows = validate_readings(readings, artifacts, now=now)
    frame = pd.DataFrame(rows, columns=["timestamp", "sensor_type", "value"])
    wide = frame.pivot(index="timestamp", columns="sensor_type", values="value").sort_index()
    return wide[[c for c in ALLOWED_CHANNELS if c in wide.columns]]


def _raw_to_scaled(window, artifacts, min_rows: int):
    if len(window) < max(1, min_rows):
        raise ContractError("INSUFFICIENT_DATA", f"need at least {min_rows} cycles, got {len(window)}")
    scale_columns = artifacts["remaining_scale_columns"]
    missing = [column for column in scale_columns if column not in window.columns]
    if missing:
        raise ContractError("VALIDATION_ERROR", f"window is missing channels: {missing}")
    raw = window[scale_columns].astype("float64")
    if not raw.notna().all().all():
        raise ContractError("VALIDATION_ERROR", "window contains missing values")
    scaled = pd.DataFrame(artifacts["scaler"].transform(raw), columns=scale_columns)
    return scaled, scale_columns


def _row_to_feature_vector(scaled_slice, scale_columns, sensor_columns, order, window_size=WINDOW):
    last_row = scaled_slice.iloc[-1]
    features = {column: float(last_row[column]) for column in scale_columns}
    features.update(compute_window_features(scaled_slice, sensor_columns, window_size))
    return {column: features[column] for column in order}


def raw_window_to_features(window, artifacts, min_rows=1, window_size=WINDOW):
    """Turn the last raw cycles of ONE machine into the model's feature vector.

    `window` is a DataFrame with one row per cycle (oldest first) holding raw, unscaled
    channels (at least the required channels; extra columns are ignored). Only the last
    `WINDOW` rows are used. The result is an ordered dict whose keys follow
    `artifacts["model_feature_columns"]` (same order as artifacts/feature_order.json).
    """
    trimmed = window.tail(window_size)
    scaled, scale_columns = _raw_to_scaled(trimmed, artifacts, min_rows=min_rows)
    sensor_columns = get_sensor_columns(scaled)
    order = artifacts["model_feature_columns"]
    return _row_to_feature_vector(scaled, scale_columns, sensor_columns, order, window_size=window_size)


def raw_window_to_sequence(window, artifacts, seq_len=WINDOW, window_size=WINDOW, **kwargs):
    """Build a (seq_len, num_features) float32 sequence for DL models (LSTM, GRU, 1D-CNN).

    Each step in the sequence uses a full `window_size` rolling history, so the API
    must provide at least `seq_len + window_size - 1` (59) raw cycles to match
    offline training features exactly.
    """
    if "window" in kwargs and kwargs["window"] is not None:
        window_size = kwargs["window"]
    required_rows = seq_len + window_size - 1
    trimmed = window.tail(required_rows)
    scaled, scale_columns = _raw_to_scaled(trimmed, artifacts, min_rows=required_rows)
    sensor_columns = get_sensor_columns(scaled)
    order = artifacts["model_feature_columns"]
    steps = []
    for end_idx in range(window_size, required_rows + 1):
        slice_df = scaled.iloc[end_idx - window_size : end_idx]
        vec = _row_to_feature_vector(slice_df, scale_columns, sensor_columns, order, window_size=window_size)
        steps.append([vec[column] for column in order])
    return np.asarray(steps, dtype=np.float32)


def out_of_training_range(window, artifacts, tolerance=0.5):
    """Names of channels in the latest row outside the training range widened by `tolerance` x range.

    Informational only: scaled values outside [0, 1] are expected for unseen engines.
    """
    scaler = artifacts["scaler"]
    last = window.iloc[-1]
    flagged = []
    for column, low, high in zip(artifacts["remaining_scale_columns"], scaler.data_min_, scaler.data_max_):
        margin = (high - low) * tolerance
        if not (low - margin <= float(last[column]) <= high + margin):
            flagged.append(column)
    return flagged
