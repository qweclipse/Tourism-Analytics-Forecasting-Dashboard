# src/preprocessing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pandas.api.types import is_numeric_dtype


def add_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Добавляет год, месяц и sin/cos месяца.
    """
    df = df.copy()
    if date_col not in df.columns:
        return df

    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_log_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Лог-трансформация числовых колонок.
    Нечисловые/дата-колонки игнорируем, чтобы не было ошибок.
    """
    df = df.copy()
    for col in cols:
        if col in df.columns and is_numeric_dtype(df[col]):
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
    return df


def add_binning(
    df: pd.DataFrame, col: str, bins: int = 3, labels: list[str] | None = None
) -> pd.DataFrame:
    """
    Квантили (binning) для непрерывной переменной.
    """
    df = df.copy()
    if col not in df.columns or not is_numeric_dtype(df[col]):
        return df

    if labels is None:
        labels = [f"bin_{i}" for i in range(bins)]

    df[f"{col}_bin"] = pd.qcut(
        df[col], q=bins, labels=labels, duplicates="drop"
    )
    return df


def build_preprocess_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Pipeline для one-hot категорий и масштабирования чисел.
    """
    transformers = []
    if categorical_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            )
        )
    if numeric_features:
        transformers.append(
            (
                "num",
                StandardScaler(),
                numeric_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor
