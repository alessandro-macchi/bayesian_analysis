# preprocessing/splitting.py

import pandas as pd
from typing import Optional, Tuple

def train_test_split_time(
    df: pd.DataFrame,
    date_col: Optional[str] = "date",
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time series dataset into training and test sets chronologically.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset (must contain a date column or DatetimeIndex).
    date_col : str, optional
        Name of the datetime column if not already index.
    train_ratio : float, default=0.8
        Proportion of data to use for training.

    Returns
    -------
    train_df : pd.DataFrame
        Training subset (first 80% of time).
    test_df : pd.DataFrame
        Test subset (last 20% of time).
    """
    out = df.copy()

    # Ensure DatetimeIndex
    if not isinstance(out.index, pd.DatetimeIndex):
        if date_col is None or date_col not in out.columns:
            raise ValueError("Provide a valid 'date_col' or set a DatetimeIndex.")
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col]).set_index(date_col)

    out = out.sort_index()

    # Compute split point
    split_idx = int(len(out) * train_ratio)
    train_df = out.iloc[:split_idx].copy()
    test_df = out.iloc[split_idx:].copy()

    print(f"[Info] Training set: {train_df.shape[0]} obs | Test set: {test_df.shape[0]} obs")

    return train_df, test_df
