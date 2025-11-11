import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Sequence, Union, Dict
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

DateLike = Union[str, pd.Timestamp]

# -----------------------------
# Helpers
# -----------------------------
def _ensure_dt_index(df: pd.DataFrame, date_col: Optional[str] = "date") -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if date_col is None or date_col not in out.columns:
            raise ValueError("Provide a valid 'date_col' or set a DatetimeIndex on the DataFrame.")
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col]).set_index(date_col)
    return out.sort_index()


def _select_series(df: pd.DataFrame, variables: Sequence[str]) -> Dict[str, pd.Series]:
    series = {}
    for v in variables:
        if v not in df.columns:
            raise KeyError(f"Column '{v}' not found in DataFrame.")
        series[v.upper()] = pd.to_numeric(df[v], errors="coerce").dropna()
    return series


# -----------------------------
# API
# -----------------------------
def plot_time_series(
    df: pd.DataFrame,
    variables: Sequence[str] = ("eui", "gpr", "cpu", "oil_price"),
    date_col: Optional[str] = "date",
    *,
    figsize: tuple = (10, 3)
) -> None:
    """
    Plot each requested variable as its own time-series figure.
    """
    out = _ensure_dt_index(df, date_col)
    S = _select_series(out, variables)

    for name, s in S.items():
        ax = s.plot(figsize=figsize, lw=1.5)
        ax.set_title(f"{name} - Time Series")
        ax.set_xlabel("Date")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_acf_for_each(
    df: pd.DataFrame,
    variables: Sequence[str] = ("eui", "gpr", "cpu", "oil_price"),
    date_col: Optional[str] = "date",
    *,
    lags: int = 36,
    figsize: tuple = (8, 4)
) -> None:
    """
    Compute and plot ACF for each variable in separate figures.
    """
    out = _ensure_dt_index(df, date_col)
    S = _select_series(out, variables)

    for name, s in S.items():
        fig = plt.figure(figsize=figsize)
        plot_acf(s, ax=plt.gca(), lags=lags, title=f"{name} - ACF")
        plt.tight_layout()
        plt.show()


def plot_pacf_for_each(
    df: pd.DataFrame,
    variables: Sequence[str] = ("eui", "gpr", "cpu", "oil_price"),
    date_col: Optional[str] = "date",
    *,
    lags: int = 36,
    method: str = "ywm",
    figsize: tuple = (8, 4)
) -> None:
    """
    Compute and plot PACF for each variable in separate figures.
    """
    out = _ensure_dt_index(df, date_col)
    S = _select_series(out, variables)

    for name, s in S.items():
        fig = plt.figure(figsize=figsize)
        plot_pacf(s, ax=plt.gca(), lags=lags, method=method, title=f"{name} - PACF")
        plt.tight_layout()
        plt.show()


def visualize_all(
    df: pd.DataFrame,
    variables: Sequence[str] = ("eui", "gpr", "cpu", "oil_price"),
    date_col: Optional[str] = "date",
    *,
    lags: int = 36
) -> None:
    """
    Convenience wrapper: for each variable, show Time Series, ACF, and PACF (three separate figures).
    """
    out = _ensure_dt_index(df, date_col)
    S = _select_series(out, variables)

    for name, s in S.items():
        # Time series
        ax = s.plot(figsize=(10, 3), lw=1.5)
        ax.set_title(f"{name} - Time Series")
        ax.set_xlabel("Date")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # ACF
        fig = plt.figure(figsize=(8, 4))
        plot_acf(s, ax=plt.gca(), lags=lags, title=f"{name} - ACF")
        plt.tight_layout()
        plt.show()

        # PACF
        fig = plt.figure(figsize=(8, 4))
        plot_pacf(s, ax=plt.gca(), lags=lags, title=f"{name} - PACF")
        plt.tight_layout()
        plt.show()
