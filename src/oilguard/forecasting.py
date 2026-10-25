from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


@dataclass
class ForecastRun:
    name: str
    order: tuple[int, int, int] | str
    start_pos: int
    pred: np.ndarray
    resid: np.ndarray
    fit_seconds: float
    aic: float
    converged: bool
    n_train: int


@dataclass
class OrderSelection:
    order: tuple[int, int, int]
    adf_pvalue: float
    aic: float
    fit_seconds: float
    converged: bool
    result: Any
    candidates: pd.DataFrame


def _converged(result: Any) -> bool:
    retvals = getattr(result, "mle_retvals", None)
    if isinstance(retvals, dict):
        return bool(retvals.get("converged", True))
    return True


def _adf_pvalue(series: pd.Series) -> float:
    values = pd.Series(series).dropna().to_numpy(dtype=float)
    if len(values) < 20 or np.nanstd(values) == 0:
        return 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(adfuller(values, autolag="AIC")[1])
    except Exception:
        return 1.0


def select_arima_order(
    train_y: pd.Series,
    max_p: int,
    max_q: int,
    max_order: int,
    d_max: int,
    trend: str = "n",
) -> OrderSelection:
    adf_p = _adf_pvalue(train_y)
    d = 1 if adf_p > 0.05 else 0
    d = min(d, d_max)
    rows: list[dict[str, Any]] = []
    best_result: Any = None
    best_order: tuple[int, int, int] | None = None
    best_aic = np.inf
    best_seconds = np.nan
    best_converged = False

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p + q > max_order:
                continue
            order = (p, d, q)
            start = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = ARIMA(
                        train_y,
                        order=order,
                        trend=trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit()
                elapsed = time.perf_counter() - start
                aic = float(result.aic)
                converged = _converged(result)
                rows.append(
                    {
                        "p": p,
                        "d": d,
                        "q": q,
                        "aic": aic,
                        "fit_seconds": elapsed,
                        "converged": converged,
                        "error": "",
                    }
                )
                if np.isfinite(aic) and aic < best_aic:
                    best_aic = aic
                    best_result = result
                    best_order = order
                    best_seconds = elapsed
                    best_converged = converged
            except Exception as exc:
                elapsed = time.perf_counter() - start
                rows.append(
                    {
                        "p": p,
                        "d": d,
                        "q": q,
                        "aic": np.nan,
                        "fit_seconds": elapsed,
                        "converged": False,
                        "error": str(exc),
                    }
                )

    if best_result is None:
        order = (1, 1, 1)
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_result = ARIMA(
                train_y,
                order=order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
        best_seconds = time.perf_counter() - start
        best_aic = float(best_result.aic)
        best_converged = _converged(best_result)
        best_order = order
        rows.append(
            {
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "aic": best_aic,
                "fit_seconds": best_seconds,
                "converged": best_converged,
                "error": "fallback",
            }
        )

    assert best_order is not None
    return OrderSelection(
        order=best_order,
        adf_pvalue=adf_p,
        aic=best_aic,
        fit_seconds=best_seconds,
        converged=best_converged,
        result=best_result,
        candidates=pd.DataFrame(rows),
    )


def rolling_forecast_arima(
    fitted_result: Any,
    y: np.ndarray,
    start_pos: int,
    fit_seconds: float,
    aic: float,
    converged: bool,
    n_train: int,
) -> ForecastRun:
    result = fitted_result
    preds = np.empty(len(y) - start_pos, dtype=float)
    for j, i in enumerate(range(start_pos, len(y))):
        pred = float(result.forecast(1).iloc[0])
        preds[j] = pred
        result = result.append([float(y[i])])
    actual = y[start_pos:]
    resid = actual - preds
    return ForecastRun(
        name="arima",
        order=tuple(fitted_result.model.order),
        start_pos=start_pos,
        pred=preds,
        resid=resid,
        fit_seconds=fit_seconds,
        aic=aic,
        converged=converged,
        n_train=n_train,
    )


def rolling_forecast_sarimax(
    train_y: pd.Series,
    train_X: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    start_pos: int,
    order: tuple[int, int, int],
    trend: str = "n",
) -> ForecastRun:
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = SARIMAX(
            train_y,
            exog=train_X,
            order=order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    fit_seconds = time.perf_counter() - start
    preds = np.empty(len(y) - start_pos, dtype=float)
    for j, i in enumerate(range(start_pos, len(y))):
        exog_step = X[i : i + 1]
        pred = float(result.forecast(1, exog=exog_step).iloc[0])
        preds[j] = pred
        result = result.append([float(y[i])], exog=exog_step)
    actual = y[start_pos:]
    resid = actual - preds
    return ForecastRun(
        name="sarimax",
        order=order,
        start_pos=start_pos,
        pred=preds,
        resid=resid,
        fit_seconds=fit_seconds,
        aic=float(result.aic),
        converged=_converged(result),
        n_train=len(train_y),
    )


def rolling_naive(y: np.ndarray, start_pos: int) -> ForecastRun:
    preds = y[start_pos - 1 : -1].astype(float)
    actual = y[start_pos:]
    resid = actual - preds
    return ForecastRun(
        name="naive",
        order="persistence",
        start_pos=start_pos,
        pred=preds,
        resid=resid,
        fit_seconds=0.0,
        aic=np.nan,
        converged=True,
        n_train=start_pos,
    )
