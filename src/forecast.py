# src/forecast.py

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def prepare_time_series(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
) -> pd.Series:
    """
    Подготовка временного ряда: индекс = дата, freq = MS.
    """
    ts = df[[date_col, target_col]].dropna()
    ts = ts.sort_values(date_col)
    ts = ts.set_index(date_col)[target_col].asfreq("MS")
    ts = ts.interpolate()
    return ts


def train_arima(ts: pd.Series, order=(1, 1, 1)):
    model = ARIMA(ts, order=order)
    fitted = model.fit()
    return fitted


def forecast_future(fitted, steps: int = 12) -> pd.Series:
    forecast = fitted.get_forecast(steps=steps)
    pred = forecast.predicted_mean
    return pred
