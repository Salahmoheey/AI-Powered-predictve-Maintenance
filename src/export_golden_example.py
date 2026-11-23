"""Write artifacts/golden_example.json: a raw window plus the exact expected feature vector.

The expected values are read from the OFFLINE pipeline output (data/features/test_features.csv),
so they are independent of serving.py. Run after preprocessing.py and features.py:

    python src/export_golden_example.py
"""
from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "test_FD001.txt"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "test_features.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
COLUMNS = (["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
           + [f"sensor_{i}" for i in range(1, 22)])
UNIT_ID = 1
WINDOW = 30


def main():
    """Build the golden example for the last WINDOW cycles of one test engine."""
    for path in (RAW_PATH, FEATURES_PATH, ARTIFACT_DIR / "feature_order.json"):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run preprocessing.py and features.py first")
    raw = pd.read_csv(RAW_PATH, sep=r"\s+", header=None, names=COLUMNS)
    features = pd.read_csv(FEATURES_PATH)
    order = json.loads((ARTIFACT_DIR / "feature_order.json").read_text(encoding="utf-8"))
    engine = raw[raw["unit_id"] == UNIT_ID].reset_index(drop=True)
    window = engine.tail(WINDOW)
    last_cycle = int(window["cycle"].iloc[-1])
    expected = features[(features["unit_id"] == UNIT_ID) & (features["cycle"] == last_cycle)].iloc[0]
    golden = {
        "description": ("Last 30 raw cycles of FD001 test engine 1 and the model feature vector the "
                        "pipeline produces for the final cycle. Expected values come from the offline "
                        "pipeline (data/features/test_features.csv), not from serving.py. A backend "
                        "implementation must reproduce them within `tolerance`."),
        "unit_id": UNIT_ID,
        "window_cycles": [int(window["cycle"].iloc[0]), last_cycle],
        "tolerance": 1e-9,
        "feature_order": order,
        "raw_window": [
            {column: (int(row[column]) if column in ("unit_id", "cycle") else float(row[column]))
             for column in COLUMNS}
            for _, row in window.iterrows()
        ],
        "expected_features": {column: float(expected[column]) for column in order},
    }
    (ARTIFACT_DIR / "golden_example.json").write_text(json.dumps(golden, indent=1), encoding="utf-8")
    print("Wrote", ARTIFACT_DIR / "golden_example.json")


if __name__ == "__main__":
    main()
