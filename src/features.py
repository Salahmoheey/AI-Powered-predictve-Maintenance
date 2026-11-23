"""Offline and online feature engineering for FD001 sensor streams.

Design rule: every engineered feature of a row is a function of the LAST `WINDOW`
cycles of the same engine (rolling mean, rolling std, and a window-limited EWMA).
Training (offline) and serving (online) therefore produce identical values, and the
backend only needs the latest `WINDOW` readings of a machine, not its full history.
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "features"
WINDOW = 30
ID_COLUMNS = ["unit_id", "cycle"]
TARGET_COLUMN = "RUL"
NON_FEATURE_COLUMNS = {"unit_id", "cycle", TARGET_COLUMN}


def get_sensor_columns(data):
    """Return sensor columns in their existing dataframe order."""
    return [column for column in data.columns if column.startswith("sensor_")]


def engineered_feature_names(sensor_columns):
    """Names of the engineered features, in the order they are produced (3 per sensor)."""
    names = []
    for column in sensor_columns:
        names += [f"{column}_rolling_mean", f"{column}_rolling_std", f"{column}_ewma"]
    return names


def model_feature_columns(base_columns):
    """Ordered model input columns: kept settings/sensors first, then engineered features.

    `base_columns` is the processed column list (it may contain unit_id, cycle and RUL,
    which are never model inputs).
    """
    sensors = [column for column in base_columns if column.startswith("sensor_")]
    base = [column for column in base_columns if column not in NON_FEATURE_COLUMNS]
    return base + engineered_feature_names(sensors)


def windowed_ewma(values, window=WINDOW):
    """EWMA (span=window, adjust=False) computed over `values` only (the last <= window rows).

    Equivalent to pandas `Series.ewm(span=window, adjust=False).mean()` evaluated on the
    window alone: the first element of the window seeds the average.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 1:
        return float(values[0])
    alpha = 2.0 / (window + 1.0)
    weights = alpha * (1.0 - alpha) ** np.arange(n - 1, -1, -1)
    weights[0] = (1.0 - alpha) ** (n - 1)
    return float(np.dot(weights, values))


def compute_window_features(scaled_window, sensors, window=WINDOW):
    """Features of the LAST row of `scaled_window` (rows ordered by cycle, scaled values)."""
    tail = scaled_window.tail(window)
    output = {}
    for column in sensors:
        values = tail[column].to_numpy(dtype=float)
        output[f"{column}_rolling_mean"] = float(values.mean())
        output[f"{column}_rolling_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output[f"{column}_ewma"] = windowed_ewma(values, window)
    return output


def create_features(data, window=WINDOW):
    """Add rolling mean/std and window-limited EWMA features independently for each engine."""
    if window < 2:
        raise ValueError("window must be at least 2")
    required = {"unit_id", "cycle"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(data.columns))}")
    result = data.sort_values(ID_COLUMNS).reset_index(drop=True).copy()
    sensors = get_sensor_columns(result)
    if not sensors:
        raise ValueError("No sensor columns found")
    grouped = result.groupby("unit_id", sort=False)
    for column in sensors:
        result[f"{column}_rolling_mean"] = grouped[column].transform(
            lambda values: values.rolling(window=window, min_periods=1).mean())
        result[f"{column}_rolling_std"] = grouped[column].transform(
            lambda values: values.rolling(window=window, min_periods=2).std().fillna(0))
        result[f"{column}_ewma"] = grouped[column].transform(
            lambda values: values.rolling(window=window, min_periods=1).apply(
                lambda chunk: windowed_ewma(chunk, window), raw=True))
    return result


def create_windows(data, window=WINDOW, target_column="RUL", group_column="unit_id"):
    """Create fixed-length sequences per engine; label each sequence by its final cycle."""
    if window < 1:
        raise ValueError("window must be positive")
    if group_column not in data or target_column not in data:
        raise ValueError(f"Data must contain {group_column!r} and {target_column!r}")
    ordered = data.sort_values([group_column, "cycle"]).reset_index(drop=True)
    excluded = {group_column, "cycle", target_column}
    feature_columns = [column for column in ordered.columns if column not in excluded]
    sequences, labels, metadata = [], [], []
    for unit_id, group in ordered.groupby(group_column, sort=False):
        values = group[feature_columns].to_numpy(dtype="float32")
        targets = group[target_column].to_numpy(dtype="float32")
        cycles = group["cycle"].to_numpy()
        for end in range(window - 1, len(group)):
            start = end - window + 1
            sequences.append(values[start:end + 1])
            labels.append(targets[end])
            metadata.append({group_column: int(unit_id), "start_cycle": int(cycles[start]),
                             "end_cycle": int(cycles[end])})
    if not sequences:
        return (pd.DataFrame(columns=[group_column, "start_cycle", "end_cycle"]),
                np.empty((0, window, len(feature_columns)), dtype="float32"),
                pd.Series(dtype="float32", name=target_column))
    return pd.DataFrame(metadata), np.stack(sequences), pd.Series(labels, name=target_column)


def validate_sensor_reading(reading, expected_columns):
    """Validate one raw sensor record before it enters online preprocessing."""
    missing = set(expected_columns) - set(reading)
    if missing:
        raise ValueError(f"Missing sensor fields: {sorted(missing)}")
    cleaned = {}
    for column in expected_columns:
        try:
            value = float(reading[column])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must be numeric") from exc
        if not pd.notna(value) or value in (float("inf"), float("-inf")):
            raise ValueError(f"{column} must be finite")
        cleaned[column] = value
    if int(reading["unit_id"]) <= 0 or int(reading["cycle"]) <= 0:
        raise ValueError("unit_id and cycle must be positive integers")
    return cleaned


def build_online_features(reading, history, artifacts, window=WINDOW):
    """Scale one raw reading and derive its current rolling features using engine history.

    `history` holds prior *scaled* rows for this engine (only the last `window - 1` rows
    are needed). Returns (feature_row, new_history) where `new_history` is already trimmed
    to the last `window` rows.

    Contract for the backend:
    - `reading` must include `unit_id` and `cycle` plus every kept raw channel. The replay
      client does not send `unit_id` (it sends `machine_id` in the URL), so the API must
      add one (any stable integer per machine) before calling.
    - Prefer `serving.raw_window_to_features`, which rebuilds the features from the last
      `window` stored readings and needs no per-engine state.
    """
    scaler = artifacts["scaler"]
    dropped = set(artifacts["dropped_columns"])
    scale_columns = artifacts["remaining_scale_columns"]
    feature_columns = artifacts["feature_columns"]
    raw_columns = [column for column in feature_columns if column not in NON_FEATURE_COLUMNS]
    validated = validate_sensor_reading(reading, [c for c in raw_columns if c not in dropped])
    row = {"unit_id": int(reading["unit_id"]), "cycle": int(reading["cycle"])}
    row.update(validated)
    scaled_values = scaler.transform(pd.DataFrame([[row[c] for c in scale_columns]], columns=scale_columns))[0]
    for column, value in zip(scale_columns, scaled_values):
        row[column] = float(value)
    current = pd.DataFrame([row])
    prior = history.copy() if history is not None and not history.empty else pd.DataFrame(columns=current.columns)
    if not prior.empty and int(current.iloc[0]["cycle"]) <= int(prior["cycle"].max()):
        raise ValueError("New cycle must be greater than the last stored cycle for this engine")
    combined = current.copy() if prior.empty else pd.concat([prior, current], ignore_index=True)
    if combined["cycle"].duplicated().any() or (len(combined) > 1 and not combined["cycle"].is_monotonic_increasing):
        raise ValueError("Cycles for an engine must be unique and strictly increasing")
    output = current.iloc[0].to_dict()
    output.update(compute_window_features(combined, get_sensor_columns(combined), window))
    return output, combined.tail(window).reset_index(drop=True)


def main():
    """Generate train, validation, and test feature CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):
        input_path = INPUT_DIR / f"{name}_processed.csv"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing {input_path}; run preprocessing.py first")
        frame = pd.read_csv(input_path)
        featured = create_features(frame)
        featured.to_csv(OUTPUT_DIR / f"{name}_features.csv", index=False)
        print(f"{name}: {featured.shape}")
    print("Feature engineering completed")


if __name__ == "__main__":
    main()
