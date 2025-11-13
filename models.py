# models.py
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _ensure_dt_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Make sure we have a DatetimeIndex, sorted ascending.
    Works with the output of your preprocessing pipeline.
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if date_col not in out.columns:
            raise ValueError(f"date_col='{date_col}' not found in columns and index is not DatetimeIndex.")
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col]).set_index(date_col)
    return out.sort_index()


# ---------------------------------------------------------------------
# 1. RANDOM WALK WITH DRIFT
# ---------------------------------------------------------------------

def fit_rw_with_drift(
    y: pd.Series,
) -> Dict[str, Any]:
    """
    Fit a Random Walk with Drift on the *level* series y_t.

        y_t = mu + y_{t-1} + eps_t

    Drift mu is estimated as the mean of first differences.
    """
    y = y.dropna()
    dy = y.diff().dropna()
    mu_hat = dy.mean()
    last_value = y.iloc[-1]

    return {
        "mu": mu_hat,
        "last_value": last_value,
        "last_index": y.index[-1],
        "name": y.name,
    }


def forecast_rw_with_drift(
    fit_result: Dict[str, Any],
    steps: int,
    freq: Optional[str] = None,
    index: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """
    Produce 'steps'-ahead forecasts from a fitted RW with drift.

    If `index` is provided, it will be used as forecast index.
    Otherwise, a DatetimeIndex is built from last_index + freq.
    """
    mu = fit_result["mu"]
    last_value = fit_result["last_value"]
    name = fit_result.get("name", "rw_drift_forecast")

    h = np.arange(1, steps + 1, dtype=float)
    fcast_values = last_value + mu * h

    if index is None:
        if freq is None:
            # Fallback: just integer index
            idx = pd.RangeIndex(start=1, stop=steps + 1)
        else:
            last_idx = fit_result["last_index"]
            idx = pd.date_range(start=last_idx, periods=steps + 1, freq=freq)[1:]
    else:
        idx = index

    return pd.Series(fcast_values, index=idx, name=name)


# ---------------------------------------------------------------------
# 2. ARIMA (possibly SARIMA if you pass seasonal_order)
# ---------------------------------------------------------------------

def fit_arima_model(
    y: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
) -> Any:
    """
    Fit an ARIMA / SARIMA model to series y_t.

    You choose (p, d, q) and seasonal (P, D, Q, m) based on ACF/PACF
    and your course slides.
    """
    y = y.dropna()
    model = ARIMA(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
    )
    res = model.fit()
    return res


def forecast_arima_model(
    res: Any,
    steps: int,
    index: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """
    Multi-step forecast from a fitted ARIMA object.
    """
    fcast_res = res.get_forecast(steps=steps)
    mean = fcast_res.predicted_mean
    if index is not None:
        mean.index = index
    return mean


# ---------------------------------------------------------------------
# 3. VAR (for your VAR-X idea, using EUI + predictors)
# ---------------------------------------------------------------------

def fit_var_system(
    df: pd.DataFrame,
    endog_cols: Sequence[str],
    lags: int = 1,
    date_col: str = "date",
) -> Any:
    """
    Fit a VAR(p) on the selected endogenous variables.

    In your application, you can set:
        endog_cols = ["eui", "gpr", "cpu", "oil_price"]

    The data are assumed *stationary* (or made so via differencing/logs).
    """
    df = _ensure_dt_index(df, date_col=date_col)
    endog = df[list(endog_cols)].dropna()

    model = VAR(endog)
    res = model.fit(maxlags=lags, ic=None, trend="c")  # you can change ic="aic" to select lags by AIC
    return res


def forecast_var_system(
    res: Any,
    steps: int,
    index: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """
    Multi-step forecast from a fitted VAR model.
    Returns a DataFrame with one column per variable.
    """
    # last k observations used as starting values
    y0 = res.y  # underlying np.array of endogenous vars
    fcast = res.forecast(y=y0, steps=steps)

    cols = res.names
    fcast_df = pd.DataFrame(fcast, columns=cols)

    if index is not None:
        fcast_df.index = index

    return fcast_df


# ---------------------------------------------------------------------
# 4. RANDOM FOREST WITH LAGGED FEATURES
# ---------------------------------------------------------------------

def build_lagged_features(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    max_lag: int = 12,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a matrix of lagged features for Random Forest.

    For each column in {target_col} ∪ feature_cols, we create lags 1..max_lag.

    Returns:
        X (DataFrame), y (Series) aligned and with NaNs dropped.
    """
    df = _ensure_dt_index(df, date_col=date_col)

    all_cols = [target_col] + list(feature_cols)
    data = {}

    for col in all_cols:
        for lag in range(1, max_lag + 1):
            data[f"{col}_lag{lag}"] = df[col].shift(lag)

    X = pd.DataFrame(data, index=df.index)
    y = df[target_col]

    # Align & drop missing (from lags)
    df_lagged = pd.concat([y, X], axis=1).dropna()
    y_out = df_lagged[target_col]
    X_out = df_lagged.drop(columns=[target_col])

    return X_out, y_out


def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    min_samples_leaf: int = 1,
) -> RandomForestRegressor:
    """
    Fit a Random Forest regressor for 1-step-ahead forecasts.
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def forecast_random_forest(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Predict using a fitted Random Forest model.
    """
    return model.predict(X_test)


# ---------------------------------------------------------------------
# Example convenience wrapper for train/test split
# ---------------------------------------------------------------------

def align_lagged_with_train_test(
    features_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    max_lag: int = 12,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Build lagged features on the *full* features_df, then
    align X,y with train/test indices from the pipeline.

    This respects the time order and uses only past information.
    """
    X_all, y_all = build_lagged_features(
        features_df,
        target_col=target_col,
        feature_cols=feature_cols,
        max_lag=max_lag,
        date_col=date_col,
    )

    # Ensure everything has DatetimeIndex
    train_df = _ensure_dt_index(train_df, date_col=date_col)
    test_df = _ensure_dt_index(test_df, date_col=date_col)

    train_idx = train_df.index.intersection(X_all.index)
    test_idx = test_df.index.intersection(X_all.index)

    X_train = X_all.loc[train_idx]
    y_train = y_all.loc[train_idx]

    X_test = X_all.loc[test_idx]
    y_test = y_all.loc[test_idx]

    return X_train, y_train, X_test, y_test
