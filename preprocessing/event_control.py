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
import pandas as pd
from typing import Optional


def add_event_controls(df: pd.DataFrame, date_col: Optional[str] = None,
                                  keep_granular: bool = False) -> pd.DataFrame:
    """
    Adds specific event controls and then aggregates them into broader macro-categories:
      1. Pandemics (SARS, COVID-19)
      2. Geopolitical & Conflict Shocks (9/11, Iraq War, Arab Spring, Russia-Ukraine, Israel-Hamas)
      3. Financial Crises (Dot-com, GFC, Eurozone Debt Crisis)
      4. Climate & Energy Disasters (Katrina, Fukushima, Deepwater Horizon, Nord Stream)
      5. Climate Policy Milestones (COP21, COP28)

    Args:
        df: Input DataFrame.
        date_col: Name of the date column if not already the index.
        keep_granular: If True, keeps all individual event columns. If False, keeps ONLY the aggregated categories.
    """
    out = _ensure_dt_index(df, date_col)

    # =========================================================
    # 1. PANDEMICS & GLOBAL HEALTH (Windows)
    # =========================================================
    # _add_window(out, "2003-02-01", "2003-07-01", "SARS_WINDOW")
    _add_window(out, "2020-02-01", "2023-05-01", "COVID_WINDOW")
    pandemic_cols = ["COVID_WINDOW"]

    # =========================================================
    # 2. GEOPOLITICAL & CONFLICT (Pulses for onset shocks)
    # =========================================================
    _add_pulse(out, "2001-09-01", "SEP11_PULSE")
    _add_pulse(out, "2003-03-01", "IRAQ_WAR_PULSE")
    _add_pulse(out, "2011-02-01", "ARAB_SPRING_LIBYA_PULSE")  # Major oil disruption
    _add_pulse(out, "2022-03-01", "RU_UA_WAR_PULSE")
    _add_pulse(out, "2023-10-01", "ISRAEL_HAMAS_PULSE")  # Red sea/Middle east shock
    conflict_cols = ["SEP11_PULSE", "IRAQ_WAR_PULSE", "ARAB_SPRING_LIBYA_PULSE",
                     "RU_UA_WAR_PULSE", "ISRAEL_HAMAS_PULSE"]

    # =========================================================
    # 3. FINANCIAL CRISES (Windows)
    # =========================================================
    _add_window(out, "2001-03-01", "2001-11-01", "DOTCOM_BUST_WINDOW")
    _add_window(out, "2007-12-01", "2009-07-01", "GFC_WINDOW")
    _add_window(out, "2010-05-01", "2012-07-01", "EURO_DEBT_CRISIS_WINDOW")
    financial_cols = ["DOTCOM_BUST_WINDOW", "GFC_WINDOW", "EURO_DEBT_CRISIS_WINDOW"]

    # =========================================================
    # 4. ENERGY DISASTERS & INFRASTRUCTURE SHOCKS (Pulses)
    # =========================================================
    _add_pulse(out, "2005-08-01", "HURRICANE_KATRINA_PULSE")  # Gulf oil shock
    _add_pulse(out, "2010-04-01", "DEEPWATER_HORIZON_PULSE")
    _add_pulse(out, "2011-03-01", "FUKUSHIMA_PULSE")
    _add_pulse(out, "2022-09-01", "NORD_STREAM_SABOTAGE_PULSE")
    disaster_cols = ["HURRICANE_KATRINA_PULSE", "DEEPWATER_HORIZON_PULSE",
                     "FUKUSHIMA_PULSE", "NORD_STREAM_SABOTAGE_PULSE"]

    # =========================================================
    # 5. CLIMATE POLICY MILESTONES (Pulses)
    # =========================================================
    _add_pulse(out, "2015-12-01", "PARIS_COP21_PULSE")
    _add_pulse(out, "2023-12-01", "COP28_FOSSIL_TRANSITION_PULSE")
    climate_policy_cols = ["PARIS_COP21_PULSE", "COP28_FOSSIL_TRANSITION_PULSE"]

    # =========================================================
    # AGGREGATION: Create the General Dummies
    # =========================================================
    # Using .max(axis=1) ensures that if any underlying event is 1, the aggregate is 1.
    out["MACRO_PANDEMIC_WINDOW"] = out[pandemic_cols].max(axis=1)
    out["MACRO_CONFLICT_PULSE"] = out[conflict_cols].max(axis=1)
    out["MACRO_FINANCIAL_WINDOW"] = out[financial_cols].max(axis=1)
    out["MACRO_ENERGY_DISASTER_PULSE"] = out[disaster_cols].max(axis=1)
    out["MACRO_CLIMATE_POLICY_PULSE"] = out[climate_policy_cols].max(axis=1)

    # Clean up: Remove granular columns if requested
    if not keep_granular:
        all_granular = (pandemic_cols + conflict_cols + financial_cols +
                        disaster_cols + climate_policy_cols)
        out = out.drop(columns=all_granular)

    return out

"""
OLD VERSION

def add_event_controls(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    out = _ensure_dt_index(df, date_col)

    # COVID-19 window (WHO PHEIC 2020-02-01 to 2023-05-01)
    _add_window(
        out,
        start="2020-02-01",
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
"""
