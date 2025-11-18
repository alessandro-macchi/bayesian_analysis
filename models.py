from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
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

def expanding_window_forecast_rw_with_drift(
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.Series:
    """
    1-step-ahead expanding-window forecasts for a Random Walk with Drift.

    At each test time t:
      - use all observations up to t-1 (train + past test) to estimate mu
      - mu_hat = mean of first differences on that window
      - forecast y_t = y_{t-1} + mu_hat

    Returns a Series indexed by y_test.index.
    """
    # Clean and build full series (train + test)
    y_train = y_train.dropna()
    y_test = y_test.dropna()

    y_full = pd.concat([y_train, y_test]).sort_index()
    full_index = y_full.index

    # Find integer position where the test part begins
    train_end_idx = y_train.index[-1]
    start_test_pos = full_index.get_loc(train_end_idx) + 1
    n_full = len(full_index)

    forecasts = []
    forecast_index = []

    for t in range(start_test_pos, n_full):
        # Expanding window up to t-1
        y_window = y_full.iloc[:t]

        # Estimate drift on this window
        dy = y_window.diff().dropna()
        mu_hat = dy.mean()

        # Last observed value in this window
        last_value = y_window.iloc[-1]

        # 1-step-ahead forecast for y at time full_index[t]
        fcast_t = last_value + mu_hat

        forecasts.append(fcast_t)
        forecast_index.append(full_index[t])

    fcast_series = pd.Series(
        forecasts,
        index=pd.Index(forecast_index, name=y_full.index.name),
        name="rw_drift_expanding",
    )

    # Align to the actual test index
    fcast_series = fcast_series.reindex(y_test.index)

    return fcast_series


# ---------------------------------------------------------------------
# 2. ARIMA (possibly SARIMA if you pass seasonal_order)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# ARIMA order selection using AIC/BIC
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2. ARIMA (fit a single model with chosen (p,d,q))
# ---------------------------------------------------------------------

def fit_arima_model(
    y: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    enforce_stationarity: bool = False,
    enforce_invertibility: bool = False,
) -> Any:
    """
    Fit an ARIMA / SARIMA model to series y_t with a *given* (p,d,q).

    Parameters
    ----------
    y : Series
        Target series.
    order : (p,d,q)
        Non-seasonal ARIMA order.
    seasonal_order : (P,D,Q,m)
        Seasonal part. Use (0,0,0,0) when there is no seasonality.
    enforce_stationarity, enforce_invertibility : bool
        As in statsmodels.ARIMA.

    Returns
    -------
    res : ARIMAResults
        Fitted model.
    """
    y_clean = y.dropna()
    model = ARIMA(
        y_clean,
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
    Multi-step forecast from a fitted ARIMA model.

    Parameters
    ----------
    res : ARIMAResults
        Fitted model from fit_arima_model.
    steps : int
        Forecast horizon.
    index : DatetimeIndex, optional
        Index to assign to the forecast series (e.g. test_df.index).

    Returns
    -------
    fcast : Series
        Forecasted values.
    """
    fcast_res = res.get_forecast(steps=steps)
    mean = fcast_res.predicted_mean
    if index is not None:
        mean.index = index
    return mean

def expanding_window_forecast_arima(
    y_train: pd.Series,
    y_test: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    enforce_stationarity: bool = False,
    enforce_invertibility: bool = False,
) -> pd.Series:
    """
    Expanding-window 1-step-ahead forecast for ARIMA.

    Given a fixed (p,d,q) and a train/test split, this function:
      - at each test time t, fits ARIMA(order) on all data up to t-1
      - produces a 1-step-ahead forecast for y_t
      - returns the full series of forecasts aligned with y_test.index

    This is the standard "expanding window" time-series cross-validation scheme.
    """
    # Concatenate to make indexing easier (but we only fit using past data)
    y_full = pd.concat([y_train, y_test])
    y_full = y_full.sort_index()

    # Indices / positions
    train_end_idx = y_train.index[-1]
    full_index = y_full.index

    # Find the integer position where the test part begins
    start_test_pos = full_index.get_loc(train_end_idx) + 1
    n_full = len(full_index)

    forecasts = []

    for t in range(start_test_pos, n_full):
        # Use all data up to t-1 as the "training window"
        y_window = y_full.iloc[:t]

        # Fit ARIMA on the current expanding window
        res = fit_arima_model(
            y=y_window,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )

        # Forecast 1-step ahead: target is y_full.index[t]
        fcast_t = forecast_arima_model(
            res=res,
            steps=1,
            index=pd.DatetimeIndex([full_index[t]]),
        )

        forecasts.append(fcast_t)

    # Concatenate all 1-step forecasts
    fcast_series = pd.concat(forecasts).sort_index()

    # Restrict to the true test period index, just in case
    fcast_series = fcast_series.reindex(y_test.index)

    fcast_series.name = f"arima_expanding_window_forecast_{order}"

    return fcast_series



# ---------------------------------------------------------------------
# 3. VAR (multivariate model for EUI + predictors)
# ---------------------------------------------------------------------

def fit_var_model(
    df: pd.DataFrame,
    endog_cols: Sequence[str],
    maxlags: int = 6,
    ic: Optional[str] = "aic",
    date_col: str = "date",
    trend: str = "c",
) -> Any:
    """
    Fit a VAR model on stationary series.

    Parameters
    ----------
    df : DataFrame
        Full dataframe (with date column or DatetimeIndex).
    endog_cols : list of str
        Names of the endogenous variables (must be stationary).
        Example: ["dlog_eui", "dlog_gpr", "dlog_cpu", "dlog_oil_price"]
    maxlags : int
        Maximum lag order to search over.
    ic : {"aic", "bic", None}
        If not None, select lag length using this information criterion.
        If None, the lag length is fixed at `maxlags`.
    trend : {"c", "nc", "ct", "ctt"}
        Trend specification (constant, none, linear, etc.)

    Returns
    -------
    res : VARResults
    """
    # ensure datetime index and sorted
    df = _ensure_dt_index(df, date_col=date_col)

    # keep only the chosen endogenous vars and drop missing values
    endog = df[list(endog_cols)].dropna()

    model = VAR(endog)

    if ic is not None:
        # Let statsmodels choose the lag order up to maxlags
        res = model.fit(maxlags=maxlags, ic=ic, trend=trend)
    else:
        # Fix lag length
        res = model.fit(maxlags, trend=trend)

    return res


def forecast_var_model(
    res: Any,
    steps: int,
    index: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """
    Multi-step forecast from a fitted VAR model.

    Parameters
    ----------
    res : VARResults
        Fitted VAR model returned by fit_var_model.
    steps : int
        Forecast horizon.
    index : DatetimeIndex, optional
        Index for the forecasted values (e.g. test_df.index).

    Returns
    -------
    fcast_df : DataFrame
        DataFrame of forecasts, columns = variable names in VAR.
    """
    # Number of lags used in the fitted VAR
    k_ar = res.k_ar

    # last k_ar observations as starting values
    y0 = res.model.endog[-k_ar:]  # shape (k_ar, n_vars)

    # forecast returns numpy array (steps x n_vars)
    fcast = res.forecast(y=y0, steps=steps)

    cols = res.names
    fcast_df = pd.DataFrame(fcast, columns=cols)

    if index is not None:
        fcast_df.index = index

    return fcast_df

def select_var_lag(
    df: pd.DataFrame,
    endog_cols: Sequence[str],
    maxlags: int = 12,
    ic: str = "bic",
    date_col: str = "date",
) -> Tuple[int, Any]:
    """
    Select VAR lag length using information criteria (AIC/BIC/HQ/FPE).

    Parameters
    ----------
    df : DataFrame
        Full dataset (with DatetimeIndex or date_col).
    endog_cols : list of str
        Names of endogenous variables (assumed stationary).
    maxlags : int
        Maximum lag order to consider.
    ic : {"aic", "bic", "hq", "fpe"}
        Criterion used to choose the lag.
    date_col : str
        Date column name if index is not DatetimeIndex.

    Returns
    -------
    best_lag : int
        Selected lag order according to the chosen IC.
    order_selection : VAROrderSelectionResults
        Full statsmodels object with criteria values.
    """
    df = df.copy()

    # ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if date_col not in df.columns:
            raise ValueError(f"date_col '{date_col}' not found in df")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
    df = df.sort_index()

    endog = df[list(endog_cols)].dropna()

    model = VAR(endog)
    order_selection = model.select_order(maxlags=maxlags)

    if ic not in {"aic", "bic", "hq", "fpe"}:
        raise ValueError("ic must be one of 'aic','bic','hq','fpe'")

    best_lag = int(getattr(order_selection, ic))

    return best_lag, order_selection

# ---------------------------------------------------------------------
# 3b. VARX via VARMAX (endogenous block + exogenous deterministic regressors)
# ---------------------------------------------------------------------

def fit_varx_model(
    df: pd.DataFrame,
    endog_cols: Sequence[str],
    exog_cols: Optional[Sequence[str]] = None,
    order: Tuple[int, int] = (1, 0),
    date_col: str = "date",
    trend: str = "c",
) -> Any:
    """
    Fit a VARX model using statsmodels VARMAX:
        y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + B x_t + u_t

    Parameters
    ----------
    df : DataFrame
        Full dataset (with date column or DatetimeIndex).
    endog_cols : list of str
        Endogenous (stochastic) variables, e.g. ["dlog_eui", "dlog_cpu", ...].
    exog_cols : list of str, optional
        Deterministic / exogenous regressors (dummies, etc.).
    order : (p, q)
        VARMAX order; here we typically use (p, 0) for VARX(p).
    date_col : str
        Date column name if index is not DatetimeIndex.
    trend : {"c", "nc", "t", "ct"}
        Trend specification (constant, none, linear, linear+constant).

    Returns
    -------
    res : VARMAXResults
        Fitted VARX model.
    """
    df_ts = _ensure_dt_index(df, date_col=date_col).copy()

    endog = df_ts[list(endog_cols)].dropna()

    exog = None
    if exog_cols is not None:
        exog = df_ts.loc[endog.index, list(exog_cols)]

    model = VARMAX(
        endog=endog,
        exog=exog,
        order=order,
        trend=trend,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    # disp=False avoids noisy optimizer output
    res = model.fit(disp=False)
    return res


def forecast_varx_model(
    res: Any,
    steps: int,
    index: Optional[pd.DatetimeIndex] = None,
    exog_future: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Multi-step forecast from a fitted VARX (VARMAX) model.

    Parameters
    ----------
    res : VARMAXResults
        Fitted model from fit_varx_model.
    steps : int
        Forecast horizon.
    index : DatetimeIndex, optional
        Index for forecasted values (e.g. test_df.index).
    exog_future : DataFrame, optional
        Future values of exogenous/deterministic regressors with
        shape (steps, n_exog). Columns must match the exog used in fitting.

    Returns
    -------
    fcast_df : DataFrame
        Forecasts for all endogenous variables.
    """
    fcast_res = res.get_forecast(steps=steps, exog=exog_future)
    mean = fcast_res.predicted_mean

    if index is not None:
        mean.index = index

    return mean



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

# ---------------------------------------------------------------------
# 4b. Random Forest time-series CV (rolling-origin)
# ---------------------------------------------------------------------

def rf_time_series_cv(
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: Sequence[Dict[str, Any]],
    initial_train_size: int,
    h: int = 1,
    step: int = 1,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Rolling-origin cross-validation for Random Forest in time series.

    Parameters
    ----------
    X, y : aligned DataFrame/Series (same index, no shuffling)
    param_grid : list of dict
        Each dict is a set of RF hyperparameters.
    initial_train_size : int
        Number of initial observations for the first training set.
    h : int
        Forecast horizon (typically h=1 for 1-step ahead).
    step : int
        Step between forecast origins.
    random_state : int
        Base random_state for reproducibility.

    Returns
    -------
    best_params : dict
        Parameter set with lowest MSFE.
    results_df : DataFrame
        One row per parameter set, columns [params, msfe].
    """
    X = X.copy()
    y = y.copy()

    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    n = len(y)
    if initial_train_size + h >= n:
        raise ValueError("initial_train_size + h must be < len(y)")

    records: List[Dict[str, Any]] = []

    for params in param_grid:
        sq_errors = []

        # rolling-origin evaluation
        for t in range(initial_train_size, n - h, step):
            train_start = 0
            train_end = t          # up to t-1
            test_start = t
            test_end = t + h       # up to t+h-1

            X_train_cv = X.iloc[train_start:train_end]
            y_train_cv = y.iloc[train_start:train_end]
            X_test_cv = X.iloc[test_start:test_end]
            y_test_cv = y.iloc[test_start:test_end]

            rf = RandomForestRegressor(
                random_state=random_state,
                n_jobs=-1,
                **params,
            )
            rf.fit(X_train_cv, y_train_cv)
            y_pred_cv = rf.predict(X_test_cv)

            errors = (y_test_cv.values - y_pred_cv) ** 2
            sq_errors.extend(errors.tolist())

        msfe = float(np.mean(sq_errors)) if sq_errors else np.inf

        records.append({
            "params": params,
            "msfe": msfe,
        })

    results_df = pd.DataFrame.from_records(records).sort_values("msfe").reset_index(drop=True)

    if results_df.empty:
        raise RuntimeError("No CV results; check your settings.")

    best_params = results_df.loc[0, "params"]

    return best_params, results_df

