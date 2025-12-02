# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


def descriptive_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].describe().T


def correlation_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].corr()


def covariance_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].cov()


def plot_hist(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(df[col].dropna(), bins=30)
    ax.set_title(f"Histogram: {col}")
    return fig


def plot_box(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.boxplot(df[col].dropna(), vert=True, labels=[col])
    ax.set_title(f"Box plot: {col}")
    return fig


def plot_heatmap(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation heatmap")
    return fig


def plot_time_series(df: pd.DataFrame, date_col: str, target_col: str):
    if date_col not in df.columns or target_col not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df[date_col], df[target_col])
    ax.set_title(f"Time series of {target_col}")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_col)
    fig.autofmt_xdate()
    return fig


def plot_decomposition(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    freq: int = 12,
):
    if date_col not in df.columns or target_col not in df.columns:
        return None

    ts = df[[date_col, target_col]].dropna()
    ts = ts.sort_values(date_col)
    ts = ts.set_index(date_col)[target_col].asfreq("MS")
    ts = ts.interpolate()

    result = seasonal_decompose(ts, model="additive", period=freq)
    fig = result.plot()
    fig.set_size_inches(6, 5)
    fig.suptitle("Time series decomposition", fontsize=12)
    return fig
