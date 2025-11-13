import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Sequence, Union, Dict
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

DateLike = Union[str, pd.Timestamp]

# -----------------------------
# Color logic
# -----------------------------

COLOR_MAP = {
    "eui": "green",
    "gpr": "red",
    "cpu": "blue",
    "oil": "orange",
}


def get_color_for_var(varname: str) -> str:
    """
    Map any variant of a variable name (e.g. log_eui, d_eui, dlog_gpr)
    to a base color family.
    """
    v = varname.lower()

    # strip common transformation prefixes
    for prefix in ("dlog_", "log_", "d_", "diff_"):
        if v.startswith(prefix):
            v = v[len(prefix):]

    # normalize oil_price, oilindex, etc. to 'oil'
    if v.startswith("oil"):
        v = "oil"

    return COLOR_MAP.get(v, "black")


def _recolor_acf_pacf(ax, color: str):
    """
    Force ACF/PACF bars, lines and CI shading on an Axes to use the given color.
    Works across statsmodels/matplotlib versions.
    """
    if ax is None:
        return

    # Bars / rectangles (main ACF/PACF bars)
    for patch in ax.patches:
        if hasattr(patch, "set_facecolor"):
            patch.set_facecolor(color)
        if hasattr(patch, "set_edgecolor"):
            patch.set_edgecolor(color)

    # Bar containers (newer matplotlib API)
    if hasattr(ax, "containers"):
        for container in ax.containers:
            for artist in container:
                if hasattr(artist, "set_facecolor"):
                    artist.set_facecolor(color)
                if hasattr(artist, "set_edgecolor"):
                    artist.set_edgecolor(color)

    # Lines (zero line, CI boundary lines, etc.)
    for line in ax.lines:
        line.set_color(color)
        if hasattr(line, "set_markerfacecolor"):
            line.set_markerfacecolor(color)
        if hasattr(line, "set_markeredgecolor"):
            line.set_markeredgecolor(color)

    # CI shading (PolyCollections)
    for coll in getattr(ax, "collections", []):
        try:
            coll.set_edgecolor(color)
            coll.set_facecolor(color)
        except Exception:
            pass


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
        # keep names as UPPER for the keys (as in your original code)
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
        color = get_color_for_var(name)
        ax = s.plot(figsize=figsize, lw=1.5, color=color)
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
        color = get_color_for_var(name)

        fig, ax = plt.subplots(figsize=figsize)
        plot_acf(s, ax=ax, lags=lags, title=f"{name} - ACF")

        _recolor_acf_pacf(ax, color)

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
        color = get_color_for_var(name)

        fig, ax = plt.subplots(figsize=figsize)
        plot_pacf(s, ax=ax, lags=lags, method=method, title=f"{name} - PACF")

        _recolor_acf_pacf(ax, color)

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
    Convenience wrapper: for each variable, show Time Series, ACF, and PACF
    (three separate figures per variable).
    """
    out = _ensure_dt_index(df, date_col)
    S = _select_series(out, variables)

    for name, s in S.items():
        color = get_color_for_var(name)

        # Time series
        ax = s.plot(figsize=(10, 3), lw=1.5, color=color)
        ax.set_title(f"{name} - Time Series")
        ax.set_xlabel("Date")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # ACF
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_acf(s, ax=ax, lags=lags, title=f"{name} - ACF")
        _recolor_acf_pacf(ax, color)
        plt.tight_layout()
        plt.show()

        # PACF
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_pacf(s, ax=ax, lags=lags, method="ywm", title=f"{name} - PACF")
        _recolor_acf_pacf(ax, color)
        plt.tight_layout()
        plt.show()
