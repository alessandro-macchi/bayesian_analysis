# preprocessing/pipeline.py

from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import os

from .import_cleaning import import_data, clean_data, apply_log_transform
from .event_control import add_event_controls
from .splitting import train_test_split_time
from .visualization import (
    plot_time_series,
    plot_acf_for_each,
    plot_pacf_for_each,
)


def preprocess_all(
    *,
    data_path: str = "data/complete_dataset.csv",
    date_col: str = "date",
    start: str = "1996-01-01",
    end: str = "2023-11-30",
    select_cols: Optional[List[str]] = None,
    rename_map: Optional[Dict[str, str]] = None,
    log_cols: Optional[List[str]] = None,
    add_event_flags: bool = True,
    train_ratio: float = 0.8,
    save_path: str = "data/preprocessed_dataset.csv",
    visualize_flags: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Orchestrates preprocessing:
      1) Import CSV
      2) Clean & filter date range (+ optional select/rename)
      3) Optional log transforms
      4) Optional event controls
      5) Save preprocessed data
      6) Train/test split
      7) Optional visualization

    Returns a dict of artifacts:
      clean_df, transformed_df, features_df, train_df, test_df
    """
    artifacts: Dict[str, Any] = {}

    # 1) Import
    df = import_data(data_path)

    # 2) Clean, filter + select/rename
    df = clean_data(
        df,
        date_col=date_col,
        start=start,
        end=end,
        select_cols=select_cols,
        rename_map=rename_map,
    )
    artifacts["clean_df"] = df

    # 3) Log transforms
    if log_cols:
        df = apply_log_transform(df, cols=log_cols)
    artifacts["transformed_df"] = df

    # 4) Event controls
    if add_event_flags:
        df = add_event_controls(df, date_col=date_col)
    artifacts["features_df"] = df

    # 5) Save the full dataset before splitting
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    print(f"[Info] Full preprocessed dataset saved to: {save_path}")

    # 6) Train/test split
    train_df, test_df = train_test_split_time(df, date_col=date_col, train_ratio=train_ratio)
    artifacts["train_df"] = train_df
    artifacts["test_df"] = test_df

    # 7) Visualization (optional)
    if visualize_flags:
        variables = visualize_flags.get("variables", ("eui", "gpr", "cpu", "oil_price"))
        lags = visualize_flags.get("lags", 36)
        do_time_series = visualize_flags.get("time_series", True)
        do_acf = visualize_flags.get("acf", True)
        do_pacf = visualize_flags.get("pacf", True)
        pacf_method = visualize_flags.get("pacf_method", "ywm")

        if do_time_series:
            plot_time_series(df, variables=variables, date_col=date_col)
        if do_acf:
            plot_acf_for_each(df, variables=variables, date_col=date_col, lags=lags)
        if do_pacf:
            plot_pacf_for_each(df, variables=variables, date_col=date_col, lags=lags, method=pacf_method)

    return artifacts
