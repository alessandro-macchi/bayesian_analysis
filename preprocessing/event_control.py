# preprocessing/event_control.py

from __future__ import annotations
import pandas as pd
from typing import Optional, Union
from datetime import datetime

# -----------------------------
# Helpers
# -----------------------------
DateLike = Union[str, datetime]

def _ensure_dt_index(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if not date_col:
            raise ValueError("Provide date_col when the DataFrame has no DatetimeIndex.")
        if date_col not in out.columns:
            raise KeyError(f"Column '{date_col}' not found.")
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        if out[date_col].isna().any():
            raise ValueError("Some dates could not be parsed; check the date column.")
        out = out.set_index(date_col)
    return out.sort_index()

def _infer_bin_width(idx: pd.DatetimeIndex) -> pd.Timedelta:
    if len(idx) < 2:
        return pd.Timedelta(days=1)
    diffs = pd.Series(idx[1:] - idx[:-1])
    med = diffs.median()
    return med if pd.notna(med) and med > pd.Timedelta(0) else pd.Timedelta(days=1)

def _add_pulse(out: pd.DataFrame, date: DateLike, colname: str) -> None:
    date = pd.to_datetime(date)
    binw = _infer_bin_width(out.index)
    mask = (out.index >= date) & (out.index < date + binw)
    out[colname] = 0
    out.loc[mask, colname] = 1

def _add_window(out: pd.DataFrame, start: DateLike, end: DateLike, colname: str) -> None:
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    binw = _infer_bin_width(out.index)
    out[colname] = ((out.index >= start) & (out.index < end + binw)).astype(int)

# -----------------------------
# Main API (specific events)
# -----------------------------
def add_event_controls(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    """
    Add event controls with the following mapping:
      - COVID-19:           WINDOW (2020-03-01 to 2023-05-01)
      - Russia-Ukraine War: PULSE  (2022-03-01)
      - Fukushima Disaster:  PULSE  (2011-03-01)
      - Global Financial Crisis: WINDOW (2007-12-01 to 2009-07-01)
      - Paris Climate Conference (COP21 adoption): PULSE (2015-12-01)
      - September 11th Terrorist Attack: PULSE (2001-09-11)
    """
    out = _ensure_dt_index(df, date_col)

    # COVID-19 window (WHO PHEIC 2020-03-01 to 2023-05-01)
    _add_window(
        out,
        start="2020-03-01",
        end="2023-05-01",
        colname="COVID_WINDOW_2020_03_11_2023_05_05",
    )

    # Russia–Ukraine war (full-scale invasion)
    _add_pulse(out, "2022-03-01", "RU_UA_WAR_PULSE_2022_02_24")

    # Fukushima nuclear disaster
    _add_pulse(out, "2011-03-01", "FUKUSHIMA_PULSE_2011_03_11")

    # Global Financial Crisis (US NBER recession window as proxy)
    _add_window(
        out,
        start="2007-12-01",
        end="2009-07-01",
        colname="GFC_WINDOW_2007_12_01_2009_06_30",
    )

    # Paris Climate Conference (COP21 agreement adoption)
    _add_pulse(out, "2015-12-01", "PARIS_COP21_PULSE_2015_12_12")

    # September 11th Terrorist Attack (2001-09-11)
    _add_pulse(out, "2001-09-01", "SEP11_PULSE_2001_09_11")

    return out
