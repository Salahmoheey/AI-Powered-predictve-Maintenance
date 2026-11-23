# Data Dictionary - NASA C-MAPSS FD001 pipeline

Source: NASA C-MAPSS FD001 (100 training engines run to failure, 100 test engines cut before failure,
one operating condition, one fault mode). Raw files are never modified: `data/raw/`.

## 1. Raw columns (26, whitespace-separated, no header)

| # | Column | Type | Meaning | Unit | Status in this project |
|---|---|---|---|---|---|
| 1 | `unit_id` | int | Engine number (1-100) | - | identifier, never a model input |
| 2 | `cycle` | int | Flight cycle of that engine (1 = first), strictly increasing | cycles | identifier / ordering, never a model input |
| 3 | `setting_1` | float | Operating setting 1 | - | **kept** |
| 4 | `setting_2` | float | Operating setting 2 (13 distinct values, near-constant) | - | **kept** |
| 5 | `setting_3` | float | Operating setting 3 (always 100) | - | **dropped** (constant) |
| 6-26 | `sensor_1` ... `sensor_21` | float | Sensor channels, see section 2 | see section 2 | 15 kept, 6 dropped (constant) |

Constant columns are detected on the **training engines only** (std <= 1e-10) and the same list is dropped
everywhere. The list is stored in `artifacts/preprocessing_config.json` (`dropped_columns`).

## 2. Sensor channels

Symbols and units follow the NASA C-MAPSS documentation (verify against the dataset readme if you cite them).

| Column | Symbol | Description | Unit | Kept? |
|---|---|---|---|---|
| `sensor_1` | T2 | Total temperature at fan inlet | degR | dropped (constant) |
| `sensor_2` | T24 | Total temperature at LPC outlet | degR | kept |
| `sensor_3` | T30 | Total temperature at HPC outlet | degR | kept |
| `sensor_4` | T50 | Total temperature at LPT outlet | degR | kept |
| `sensor_5` | P2 | Pressure at fan inlet | psia | dropped (constant) |
| `sensor_6` | P15 | Total pressure in bypass duct | psia | kept |
| `sensor_7` | P30 | Total pressure at HPC outlet | psia | kept |
| `sensor_8` | Nf | Physical fan speed | rpm | kept |
| `sensor_9` | Nc | Physical core speed | rpm | kept |
| `sensor_10` | epr | Engine pressure ratio (P50/P2) | - | dropped (constant) |
| `sensor_11` | Ps30 | Static pressure at HPC outlet | psia | kept |
| `sensor_12` | phi | Ratio of fuel flow to Ps30 | pps/psi | kept |
| `sensor_13` | NRf | Corrected fan speed | rpm | kept |
| `sensor_14` | NRc | Corrected core speed | rpm | kept |
| `sensor_15` | BPR | Bypass ratio | - | kept |
| `sensor_16` | farB | Burner fuel-air ratio | - | dropped (constant) |
| `sensor_17` | htBleed | Bleed enthalpy | - | kept |
| `sensor_18` | Nf_dmd | Demanded fan speed | rpm | dropped (constant) |
| `sensor_19` | PCNfR_dmd | Demanded corrected fan speed | rpm | dropped (constant) |
| `sensor_20` | W31 | HPT coolant bleed | lbm/s | kept |
| `sensor_21` | W32 | LPT coolant bleed | lbm/s | kept |

There is **no vibration channel** in FD001, so no FFT features are computed.

## 3. Derived columns

| Column | Type | Definition |
|---|---|---|
| `RUL` | float | Remaining useful life in cycles, capped at 125. Train: `max_cycle - cycle`. Test: `RUL_FD001[unit] + last_cycle - cycle`. Used only as a label. |
| `<sensor>_rolling_mean` | float | Mean of the scaled sensor over the last 30 cycles of the same engine (fewer at the start of an engine, minimum 1). |
| `<sensor>_rolling_std` | float | Sample standard deviation (ddof = 1) over the same window; 0 when the window has 1 row. |
| `<sensor>_ewma` | float | Exponentially weighted mean, span = 30, `adjust=False`, computed **inside the same window** (the first row of the window seeds the average). |

Scaling: `MinMaxScaler` fitted on the 80 training engines only, applied to the 17 kept channels
(`setting_1`, `setting_2` and the 15 kept sensors) before any rolling feature is computed.
Validation/test values can fall slightly outside [0, 1]; they are not clipped.

## 4. Generated datasets (not committed, rebuilt by the pipeline)

| File | Rows | Engines | Columns | Content |
|---|---|---|---|---|
| `data/processed/train_processed.csv` | 16,561 | 80 | 20 | `unit_id`, `cycle`, 2 settings, 15 scaled sensors, `RUL` |
| `data/processed/validation_processed.csv` | 4,070 | 20 | 20 | same |
| `data/processed/test_processed.csv` | 13,096 | 100 | 20 | same (RUL per row derived from `RUL_FD001`) |
| `data/features/*_features.csv` | same rows | same | 65 | the 20 columns above + 45 engineered features |

## 5. Model input vector (62 values, fixed order)

`artifacts/feature_order.json` lists the exact order: `setting_1`, `setting_2`, the 15 scaled sensors, then for each
sensor `<sensor>_rolling_mean`, `<sensor>_rolling_std`, `<sensor>_ewma`.
`unit_id`, `cycle` and `RUL` are never model inputs. For sequence models, `create_windows()` (in `src/features.py`)
returns windows of 30 consecutive cycles per engine: pass a frame from `data/features/*_features.csv` to get
sequences of shape `(N, 30, 62)`, or a frame from `data/processed/*_processed.csv` to get `(N, 30, 17)`.

## 6. Split files (`data/splits/`, committed)

| File | Content |
|---|---|
| `train_unit_ids.csv` / `validation_unit_ids.csv` | 80 / 20 engine IDs from `GroupShuffleSplit(test_size=0.2, random_state=42)` on the training file |
| `test_unit_ids.csv` | the 100 test engines (never used for fitting or tuning) |
| `group_kfold/fold_{1..5}_{train,validation}_units.csv` | `GroupKFold(5)` inside the 80 training engines |
| `split_metadata.json` | seed, validation fraction, engine counts |

## 7. Artifacts (`artifacts/`, committed)

| File | Content |
|---|---|
| `scaler.joblib` | dict with the fitted `MinMaxScaler`, `dropped_columns`, `remaining_scale_columns`, `model_feature_columns`, `rul_cap`, `window`, `sklearn_version` |
| `preprocessing_config.json` | human-readable copy, including per-channel training min/max |
| `feature_order.json` | the 62 model input columns in order |
| `golden_example.json` | raw 30-cycle window + expected feature vector, for verifying a backend implementation |

## 8. Training range of every kept channel (raw units)

Used by the scaler (`train_min`, `train_max`) and by `serving.out_of_training_range`.

| Channel | train_min | train_max |
|---|---|---|
| `setting_1` | -0.0087 | 0.0087 |
| `setting_2` | -0.0006 | 0.0006 |
| `sensor_2` | 641.21 | 644.53 |
| `sensor_3` | 1571.04 | 1616.91 |
| `sensor_4` | 1385.19 | 1441.49 |
| `sensor_6` | 21.6 | 21.61 |
| `sensor_7` | 549.85 | 556.06 |
| `sensor_8` | 2387.9 | 2388.56 |
| `sensor_9` | 9021.73 | 9244.59 |
| `sensor_11` | 46.85 | 48.52 |
| `sensor_12` | 518.69 | 523.38 |
| `sensor_13` | 2387.88 | 2388.56 |
| `sensor_14` | 8099.94 | 8293.72 |
| `sensor_15` | 8.3279 | 8.5848 |
| `sensor_17` | 388 | 400 |
| `sensor_20` | 38.14 | 39.43 |
| `sensor_21` | 22.8942 | 23.6184 |
