# preprocessing/import_cleaning.py

import pandas as pd
import numpy as np
from typing import Optional, List, Dict


def import_data(path: str = "project/data/complete_dataset.csv") -> pd.DataFrame:
    """
    Import the complete dataset from CSV and strip column names.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def clean_data(
    df: pd.DataFrame,
    date_col: str = "date",
    start: str = "1996-01-01",
    end: str = "2023-11-30",
    select_cols: Optional[List[str]] = None,
    rename_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Clean dataset by:
      - Ensuring 'date' column is datetime type
      - Checking and reporting missing values
      - Filtering data between Jan 1996 and Nov 2023
      - Selecting specific columns if requested
      - Renaming columns if requested
    """
    out = df.copy()

    # --- Ensure datetime type ---
    if date_col not in out.columns:
        raise KeyError(f"Column '{date_col}' not found in DataFrame.")
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    # --- Handle missing date values ---
    missing_dates = out[date_col].isna().sum()
    if missing_dates > 0:
        print(f"[Warning] {missing_dates} missing or unparseable dates found. Dropping them.")
        out = out.dropna(subset=[date_col])

    # --- Report general missing values ---
    total_missing = out.isna().sum()
    if total_missing.any():
        print("\n[Info] Missing values by column:")
        print(total_missing[total_missing > 0])

    # --- Filter by date range ---
    out = out.set_index(date_col).sort_index()
    mask = (out.index >= pd.to_datetime(start)) & (out.index <= pd.to_datetime(end))
    out = out.loc[mask]
    out = out[~out.index.duplicated(keep="first")]

    # --- Select subset of columns (if provided) ---
    if select_cols:
        missing_cols = [c for c in select_cols if c not in out.columns]
        if missing_cols:
            print(f"[Warning] The following columns are missing and will be skipped: {missing_cols}")
        existing_cols = [c for c in select_cols if c in out.columns]
        out = out[existing_cols]

    # --- Rename columns (if provided) ---
    if rename_map:
        out = out.rename(columns=rename_map)

    return out


def apply_log_transform(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Apply log transformation to selected numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (already cleaned and with datetime index).
    cols : list of str
        Columns to log-transform.

    Returns
    -------
    pd.DataFrame
        DataFrame with new columns 'log_<col>' added.
    """
    out = df.copy()

    for col in cols:
        if col not in out.columns:
            print(f"[Warning] Column '{col}' not found; skipping log transform.")
            continue

        # Ensure positive values
        positive_mask = out[col] > 0
        n_invalid = (~positive_mask).sum()
        if n_invalid > 0:
            print(f"[Warning] {n_invalid} non-positive values in '{col}' replaced with NaN before log.")
            out.loc[~positive_mask, col] = np.nan

        out[f"log_{col}"] = np.log(out[col])

    return out
