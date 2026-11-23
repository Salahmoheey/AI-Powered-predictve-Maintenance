"""Engine-level cross-validation utilities."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold


def get_group_kfold_splits(data, n_splits=5, group_column="unit_id"):
    """Return GroupKFold indices so one engine never appears in both sides of a fold."""
    if group_column not in data.columns:
        raise ValueError(f"Missing group column: {group_column}")
    groups = data[group_column]
    if groups.nunique() < n_splits:
        raise ValueError(f"Need at least {n_splits} distinct engines")
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(data, groups=groups))


def save_group_kfold_unit_ids(data, output_dir, n_splits=5, group_column="unit_id"):
    """Save train/validation engine IDs for every GroupKFold split."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = get_group_kfold_splits(data, n_splits=n_splits, group_column=group_column)
    for fold_number, (train_index, validation_index) in enumerate(splits, start=1):
        train_ids = sorted(data.iloc[train_index][group_column].astype(int).unique())
        validation_ids = sorted(data.iloc[validation_index][group_column].astype(int).unique())
        pd.DataFrame({group_column: train_ids}).to_csv(output / f"fold_{fold_number}_train_units.csv", index=False)
        pd.DataFrame({group_column: validation_ids}).to_csv(output / f"fold_{fold_number}_validation_units.csv", index=False)
    return len(splits)
