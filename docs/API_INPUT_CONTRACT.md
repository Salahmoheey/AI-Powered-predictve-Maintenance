# API Input Contract - raw sensor readings

Status: proposed by the data-pipeline owner, aligned with Stage 4 section 6 (`POST /machines/{machine_id}/sensor-logs`).
The backend owner must confirm or change it; **any change must be mirrored in `src/replay.py` and `src/serving.py`**
(the tests in `tests/test_replay_contract.py` and `tests/test_serving.py` will fail if they drift apart).

## 1. Ingestion endpoint

`POST /machines/{machine_id}/sensor-logs` - header `Authorization: Bearer <JWT>` (role: engineer or admin).

```json
{
  "readings": [
    {"sensor_type": "setting_1", "value": 0.0023, "timestamp": "2026-11-21T10:00:00Z"},
    {"sensor_type": "sensor_2", "value": 643.02, "timestamp": "2026-11-21T10:00:00Z"}
  ]
}
```

| Field | Type | Rules |
|---|---|---|
| `readings` | array | 1 to 1,000 items (more -> `413 BATCH_TOO_LARGE`). The whole batch is rejected if any item is invalid. |
| `readings[].sensor_type` | string | One of the 24 raw channel names: `setting_1..3`, `sensor_1..21`. |
| `readings[].value` | number | Finite (no NaN/Infinity, not a string/bool). `sensor_*` values must be >= 0. Raw, unscaled. |
| `readings[].timestamp` | string | ISO-8601 **with** UTC offset (`Z` or `+00:00`), not in the future (5 s tolerance). |

### One cycle = one timestamp
- All channels of one engine cycle share **one timestamp**; later cycles have a **later** timestamp.
  The cycle order is recovered from the timestamp (there is no `cycle` field, and `cycle` is not a model input).
- A cycle must contain all **17 required channels**: `setting_1`, `setting_2`, `sensor_2`, `sensor_3`, `sensor_4`,
  `sensor_6`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`,
  `sensor_17`, `sensor_20`, `sensor_21`. The other 7 channels (`setting_3`, `sensor_1`, `sensor_5`, `sensor_10`,
  `sensor_16`, `sensor_18`, `sensor_19`) are constant in FD001: accept them (the replay sends all 24) but they are
  ignored by the recipe. The authoritative list is `artifacts/preprocessing_config.json`.
- Duplicate `(machine_id, timestamp, sensor_type)` -> rejected (`422`).
- Values outside the training range are **accepted**; `serving.out_of_training_range` can flag them for logging.

### Responses
| Status | When | Body |
|---|---|---|
| 201 | stored | `{"inserted": 24, "first_log_id": 101, "last_log_id": 124}` |
| 404 | unknown machine | `MACHINE_NOT_FOUND` |
| 413 | more than 1,000 readings | `BATCH_TOO_LARGE` |
| 422 | any validation rule above failed (nothing stored) | `VALIDATION_ERROR` with a message naming the item, e.g. `readings[3]: value must be a finite number` |

`serving.validate_readings()` implements exactly these rules and raises `ContractError(code, message, http_status)`.

## 2. How the replay maps engines to machines
NASA `unit_id` is **not** a backend id. `data/machine_mapping.json` (copy of `machine_mapping.example.json`) maps every
`unit_id` to a real `machine_id`. The replay posts one request per cycle (24 readings) to that machine.

## 3. Prediction endpoint (feature side)
`POST /machines/{machine_id}/predictions` (optional `window_size`; default **59** for sequence DL models `GRU`/`LSTM`/`1D-CNN`, or **30** for tabular `XGBoost`):

### A. Sequence Deep Learning Models (`GRU` — production default, `LSTM`, `1D-CNN`)
1. Read the latest `window_size = 59` **distinct timestamps** (`seq_len + window - 1 = 30 + 30 - 1`) of the machine from `SENSOR_LOGS` and pivot them to one row per timestamp (`serving.readings_to_frame`).
2. Fewer than 59 rows -> `422 INSUFFICIENT_DATA` (`raw_window_to_sequence(..., seq_len=30, window=30)`).
3. `serving.raw_window_to_sequence(window, artifacts, seq_len=30, window=30)` -> `(30, 62)` `float32` sequence (see `docs/FEATURE_RECIPE.md`).
4. Add batch dimension `(1, 30, 62)` and pass to the sequence model (`models/gru_rul.pt`).

### B. Tabular Model (`XGBoost`)
1. Read the latest `window_size = 30` **distinct timestamps** of the machine from `SENSOR_LOGS` and pivot them (`serving.readings_to_frame`).
2. Fewer than 30 rows -> `422 INSUFFICIENT_DATA` (`raw_window_to_features(..., min_rows=30)`).
3. `serving.raw_window_to_features(window, artifacts)` -> ordered 62-value vector matching `artifacts/feature_order.json`.
4. Pass the `(1, 62)` vector to `models/xgboost_rul.json`.

No per-machine state is needed: the features depend only on the last 59 cycles (for DL sequence models) or last 30 cycles (for XGBoost), which are already in `SENSOR_LOGS`.
