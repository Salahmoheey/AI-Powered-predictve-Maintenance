# Recipe: raw sensor readings -> model features

Everything below is implemented in `src/serving.py` (`raw_window_to_features` and `raw_window_to_sequence`) and verified against the offline
training pipeline. If the backend re-implements it, it must reproduce `artifacts/golden_example.json`.

## Inputs
- **For single-step tabular inference (`raw_window_to_features`):** the latest up-to-**30** cycles of one machine, oldest first, **raw** values (the 17 required channels).
- **For 30-step sequence DL inference (`raw_window_to_sequence`):** the latest **59** raw cycles of one machine (`30 + 30 - 1`), oldest first, so that every step in the 30-step output window has a full 30-cycle rolling context.
- `artifacts/scaler.joblib` (or the per-channel `train_min` / `train_max` in `artifacts/preprocessing_config.json`).
- `artifacts/feature_order.json` (the output order).

## Steps (for the newest cycle `t`)
1. **Keep** only the 17 required channels; ignore `setting_3` and the six constant sensors.
2. **Validate**: finite numbers, no missing values, timestamps strictly increasing (see `API_INPUT_CONTRACT.md`).
3. **Scale** every kept channel with the training min/max: `scaled = (raw - train_min) / (train_max - train_min)`.
   Do not clip. Do not refit the scaler.
4. **Window**: take the last `n = min(30, available cycles)` scaled rows (including the newest).
5. **Per sensor** (the 15 kept sensors, not the settings) compute over that window:
   - `<sensor>_rolling_mean` = arithmetic mean
   - `<sensor>_rolling_std` = sample std (ddof = 1); `0` when `n = 1`
   - `<sensor>_ewma` = EWMA with `alpha = 2/31`, seeded by the first row of the window:
     `y_0 = x_0`, `y_k = alpha * x_k + (1 - alpha) * y_(k-1)`, value = `y_(n-1)`
6. **Assemble** the vector in the order of `feature_order.json` (62 values):
   `setting_1`, `setting_2` (scaled, newest cycle), the 15 scaled sensors (newest cycle), then for every sensor
   in sensor order: mean, std, ewma.

`unit_id`, `cycle`, timestamps and `RUL` are not part of the vector.

## Building a 30-step sequence for Deep Learning models (`raw_window_to_sequence`)
Sequence models (`GRU`, `LSTM`, `1D-CNN`) expect an input tensor of shape `(1, 30, 62)` representing 30 consecutive cycles where each cycle's rolling features were computed over its own preceding 30 cycles:
1. Require at least `seq_len + window - 1 = 30 + 30 - 1 = 59` raw cycles (oldest first).
2. Slice the latest 59 raw rows (`tail(59)`), scale all 59 rows once using `artifacts["scaler"]`.
3. For each offset `i` in `0 .. 29`, take the 30-row slice `scaled[i : i + 30]` and compute the 62-feature vector for step `i`.
4. Stack the 30 vectors into a `float32` array of shape `(30, 62)` matching `create_windows()` in `src/features.py` with zero numerical drift.

## Worked example (engine 1, `sensor_2`, first three cycles)
Training range of `sensor_2`: min 641.21, max 644.53 (range 3.32).

| Cycle | raw | scaled |
|---|---|---|
| 1 | 643.02 | 0.545181 |
| 2 | 641.71 | 0.150602 |
| 3 | 642.46 | 0.376506 |

At cycle 3 (window = cycles 1-3): `sensor_2_rolling_mean` = 0.357430, `sensor_2_rolling_std` = 0.197980,
`sensor_2_ewma` = 0.510484 (`alpha = 0.064516`: y1 = 0.519724, y2 = 0.510484).
At cycle 1 the window has one row: mean = 0.545181, std = 0, ewma = 0.545181.

## Using the reference code
```python
from serving import load_artifacts, readings_to_frame, raw_window_to_features, raw_window_to_sequence

artifacts = load_artifacts()                                         # once at start-up
window = readings_to_frame(rows_from_sensor_logs, artifacts)         # validates + pivots

# 1) Single-step vector (62,) for XGBoost:
vector = raw_window_to_features(window, artifacts, min_rows=30)
xgb_input = list(vector.values())                                    # already in model order

# 2) 30-step sequence (30, 62) for GRU / LSTM / 1D-CNN:
seq_input = raw_window_to_sequence(window, artifacts, seq_len=30, window=30)  # requires >= 59 rows
```

## Verify an implementation
1. Load `artifacts/golden_example.json`; feed `raw_window` (30 raw cycles of engine 1) to your implementation.
2. The result must equal `expected_features` for every key, within `tolerance` (1e-9), in the order `feature_order`.
3. `pytest tests/test_serving.py` runs the same check for the reference code, plus a full comparison with the offline
   features and 30-step sequences for two test engines.

## Environment
`scaler.joblib` is pickled with scikit-learn 1.9.1 (`requirements.txt`): serve with the same version, or rebuild the
scaler from `train_min` / `train_max` in `preprocessing_config.json` (MinMax scaling is just the formula above).
