"""Load, validate, split, and scale the NASA C-MAPSS FD001 dataset."""
from pathlib import Path
import json
import joblib
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler

try:  # imported as a package (e.g. src.preprocessing)
    from .features import WINDOW, model_feature_columns
    from .validation import save_group_kfold_unit_ids
except ImportError:  # executed as a script: python src/preprocessing.py
    from features import WINDOW, model_feature_columns
    from validation import save_group_kfold_unit_ids

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "splits"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

COLUMNS = (
    ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
SCALE_COLUMNS = ["setting_1", "setting_2", "setting_3"] + SENSOR_COLUMNS
RUL_CAP = 125
CONSTANT_TOLERANCE = 1e-10
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


def load_data():
    """Read raw train, test, and test-RUL files from data/raw."""
    required = [RAW_DIR / "train_FD001.txt", RAW_DIR / "test_FD001.txt", RAW_DIR / "RUL_FD001.txt"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing NASA dataset file(s): " + ", ".join(missing))
    train = pd.read_csv(required[0], sep=r"\s+", header=None, names=COLUMNS)
    test = pd.read_csv(required[1], sep=r"\s+", header=None, names=COLUMNS)
    rul = pd.read_csv(required[2], header=None, names=["RUL"])
    return train, test, rul


def validate_data(data, name):
    """Validate schema, numeric types, finite values, ranges, uniqueness, and cycle order."""
    if list(data.columns) != COLUMNS:
        raise ValueError(f"{name}: unexpected columns or column order")
    converted = data.copy()
    for column in COLUMNS:
        converted[column] = pd.to_numeric(converted[column], errors="coerce")
    if converted.isna().any().any():
        bad = converted.columns[converted.isna().any()].tolist()
        raise ValueError(f"{name}: missing or non-numeric values in {bad}")
    if not np.isfinite(converted.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: contains infinite values")
    if (converted["unit_id"] <= 0).any() or (converted["cycle"] <= 0).any():
        raise ValueError(f"{name}: unit_id and cycle must be positive")
    negative_sensors = [column for column in SENSOR_COLUMNS if (converted[column] < 0).any()]
    if negative_sensors:
        raise ValueError(f"{name}: sensor readings must be non-negative; invalid columns: {negative_sensors}")
    if not np.allclose(converted["unit_id"], converted["unit_id"].astype(int)):
        raise ValueError(f"{name}: unit_id must contain integers")
    if not np.allclose(converted["cycle"], converted["cycle"].astype(int)):
        raise ValueError(f"{name}: cycle must contain integers")
    if converted.duplicated(["unit_id", "cycle"]).any():
        raise ValueError(f"{name}: duplicate (unit_id, cycle) records found")
    for unit_id, group in converted.groupby("unit_id", sort=False):
        cycles = group["cycle"].to_numpy()
        if len(cycles) > 1 and not np.all(np.diff(cycles) > 0):
            raise ValueError(f"{name}: cycles are not strictly increasing for unit {unit_id}")
    print(f"{name}: validation passed ({len(converted):,} rows)")
    return converted


def calculate_train_rul(train):
    """Calculate capped RUL labels for complete training trajectories."""
    result = train.copy()
    max_cycle = result.groupby("unit_id")["cycle"].transform("max")
    result["RUL"] = (max_cycle - result["cycle"]).clip(upper=RUL_CAP)
    return result


def calculate_test_rul(test, rul):
    """Attach NASA-provided end-of-test RUL labels to each test trajectory."""
    result = test.copy()
    last_cycles = result.groupby("unit_id")["cycle"].max()
    units = sorted(result["unit_id"].unique())
    if len(units) != len(rul):
        raise ValueError("Test units and RUL labels do not match")
    rul_by_unit = dict(zip(units, rul["RUL"].astype(float)))
    result["RUL"] = (
        result["unit_id"].map(rul_by_unit)
        + result["unit_id"].map(last_cycles)
        - result["cycle"]
    ).clip(upper=RUL_CAP)
    return result


def split_data(train):
    """Split complete engines into train and validation without engine leakage."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE)
    train_index, validation_index = next(splitter.split(train, groups=train["unit_id"]))
    return train.iloc[train_index].copy(), train.iloc[validation_index].copy()


def remove_constant_sensors(train, validation, test):
    """Find constant sensor/setting columns on training rows only and apply that list everywhere.

    All three operating settings and all 21 sensors are checked. In FD001 this drops six
    constant sensors and setting_3.
    """
    candidates = SENSOR_COLUMNS + ["setting_2", "setting_3"]
    std_values = train[candidates].std()
    dropped_columns = std_values[std_values.fillna(0) <= CONSTANT_TOLERANCE].index.tolist()
    remaining_scale_columns = [column for column in SCALE_COLUMNS if column not in dropped_columns]
    frames = [frame.drop(columns=dropped_columns).copy() for frame in (train, validation, test)]
    return (*frames, dropped_columns, remaining_scale_columns, std_values.to_dict())


def normalize_data(train, validation, test, remaining_scale_columns):
    """Fit MinMaxScaler on training rows and transform validation/test with the same scaler."""
    frames = [frame.copy() for frame in (train, validation, test)]
    for frame in frames:
        frame[remaining_scale_columns] = frame[remaining_scale_columns].astype("float64")
    scaler = MinMaxScaler()
    scaler.fit(frames[0][remaining_scale_columns])
    for frame in frames:
        frame.loc[:, remaining_scale_columns] = scaler.transform(frame[remaining_scale_columns])
    return (*frames, scaler)


def save_split_ids(train, validation, test):
    """Save engine IDs so every team member can reproduce the same split."""
    split_dir = SPLIT_DIR
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        ids = sorted(frame["unit_id"].astype(int).unique().tolist())
        pd.DataFrame({"unit_id": ids}).to_csv(split_dir / f"{name}_unit_ids.csv", index=False)
    metadata = {"random_state": RANDOM_STATE, "validation_size": VALIDATION_SIZE,
                "train_units": int(train["unit_id"].nunique()),
                "validation_units": int(validation["unit_id"].nunique()),
                "test_units": int(test["unit_id"].nunique())}
    (split_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_datasets():
    """Run load -> validate -> label -> split -> drop constants -> scale, without saving anything."""
    train_raw, test_raw, rul = load_data()
    train_raw = validate_data(train_raw, "Training data")
    test_raw = validate_data(test_raw, "Testing data")
    train = calculate_train_rul(train_raw)
    test = calculate_test_rul(test_raw, rul)
    train, validation = split_data(train)
    train, validation, test, dropped, scale_columns, std_values = remove_constant_sensors(train, validation, test)
    train, validation, test, scaler = normalize_data(train, validation, test, scale_columns)
    train, validation, test = [frame.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
                               for frame in (train, validation, test)]
    return {"train": train, "validation": validation, "test": test, "scaler": scaler,
            "dropped_columns": dropped, "remaining_scale_columns": scale_columns,
            "std_values": std_values}


def save_artifacts(result, output_dir=ARTIFACT_DIR):
    """Write the files the backend and the model owners need to reproduce the preprocessing.

    - scaler.joblib: fitted MinMaxScaler plus the metadata needed to apply it.
    - preprocessing_config.json: human-readable copy (dropped columns, per-column min/max).
    - feature_order.json: ordered list of the model input columns.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scaler, scale_columns = result["scaler"], result["remaining_scale_columns"]
    processed_columns = result["train"].columns.tolist()
    feature_order = model_feature_columns(processed_columns)
    artifacts = {
        "scaler": scaler,
        "dropped_columns": result["dropped_columns"],
        "remaining_scale_columns": scale_columns,
        "std_values": result["std_values"],
        "feature_columns": result["train"].drop(columns=["RUL"]).columns.tolist(),
        "model_feature_columns": feature_order,
        "rul_cap": RUL_CAP,
        "window": WINDOW,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(artifacts, output / "scaler.joblib")
    config = {
        "sklearn_version": sklearn.__version__,
        "scaler": "MinMaxScaler fitted on the 80 training engines only",
        "rul_cap": RUL_CAP,
        "window": WINDOW,
        "random_state": RANDOM_STATE,
        "dropped_columns": result["dropped_columns"],
        "scaled_columns": scale_columns,
        "scaled_column_ranges": {
            column: {"train_min": float(low), "train_max": float(high)}
            for column, low, high in zip(scale_columns, scaler.data_min_, scaler.data_max_)
        },
        "model_input_columns": feature_order,
    }
    (output / "preprocessing_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "feature_order.json").write_text(json.dumps(feature_order, indent=2), encoding="utf-8")


def main():
    """Run the complete preprocessing pipeline and persist its artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = build_datasets()
    train, validation, test = result["train"], result["validation"], result["test"]
    save_split_ids(train, validation, test)
    save_group_kfold_unit_ids(train, SPLIT_DIR / "group_kfold", n_splits=5)
    train.to_csv(OUTPUT_DIR / "train_processed.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation_processed.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test_processed.csv", index=False)
    save_artifacts(result)
    print("\nPreprocessing completed")
    print("Dropped columns:", result["dropped_columns"])
    print("Train/validation/test:", train.shape, validation.shape, test.shape)
    print("Artifacts written to:", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
