# src/models.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from .preprocessing import build_preprocess_pipeline


def train_regression_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    categorical_cols: list[str] | None = None,
):
    """
    Обучает RandomForestRegressor, возвращает (pipeline, metrics).
    """
    if categorical_cols is None:
        categorical_cols = []

    df = df.dropna(subset=[target_col])

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_features = [
        c for c in feature_cols if c not in categorical_cols
    ]
    cat_features = [c for c in feature_cols if c in categorical_cols]

    preprocessor = build_preprocess_pipeline(
        numeric_features=numeric_features,
        categorical_features=cat_features,
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": float(mae),
        "R2": float(r2),
    }

    return pipe, metrics
