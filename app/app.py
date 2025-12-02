# app/app.py

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandas.errors import ParserError

from src.config import DATA_DIR
from src.data_loader import load_data
from src.preprocessing import add_date_features, add_log_transform, add_binning
from src.eda import (
    descriptive_stats,
    correlation_matrix,
    covariance_matrix,
    plot_hist,
    plot_box,
    plot_heatmap,
    plot_time_series,
    plot_decomposition,
)
from src.models import train_regression_model
from src.forecast import prepare_time_series, train_arima, forecast_future

def build_wide_time_series(df_raw: pd.DataFrame, id_col: str, id_value: str):
    """
    Делает из горизонтальной таблицы (годы в колонках) нормальный
    временной ряд: date / value для выбранной строки.

    df_raw  – исходный DataFrame (как прочитан из CSV)
    id_col  – колонка идентификатора (страна/регион)
    id_value – значение идентификатора (конкретная страна/ряд)
    """
    # колонки, похожие на годы: '1960', '1975', ...
    year_cols = [
        c for c in df_raw.columns
        if str(c).strip().isdigit() and len(str(c).strip()) == 4
    ]
    if not year_cols:
        return None, year_cols

    row = df_raw[df_raw[id_col] == id_value]
    if row.empty:
        return None, year_cols

    row = row.iloc[0]

    ts_df = pd.DataFrame({
        "date": pd.to_datetime(pd.Series(year_cols).astype(str) + "-01-01"),
        "value": pd.to_numeric(
            pd.Series([row[c] for c in year_cols]),
            errors="coerce"
        ),
    })

    ts_df = ts_df.dropna(subset=["value"])
    return ts_df, year_cols

st.set_page_config(
    page_title="Tourism Analytics Dashboard",
    layout="wide",
)

st.title("📊 Tourism Analytics & Forecasting")

# ---------------- Sidebar: загрузка данных ----------------

st.sidebar.header("Data")

data_mode = st.sidebar.radio(
    "Data source",
    ["From /data folder", "Upload CSV"],
)

df_raw: pd.DataFrame | None = None

# ---- Загрузка пользователем ----
if data_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    sep_mode = st.sidebar.selectbox(
        "Delimiter",
        options=["auto", ",", ";", "tab (\\t)", "|", "space", "custom"],
        index=0,
    )

    custom_sep = ""
    if sep_mode == "custom":
        custom_sep = st.sidebar.text_input(
            "Custom delimiter (1 символ)",
            value=";",
            max_chars=5,
        )

    skip_rows_upload = st.sidebar.number_input(
        "Skip first N rows (headers/notes)",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
    )

    save_uploaded = st.sidebar.checkbox(
        "Save uploaded CSV into /data",
        value=False,
    )

    if uploaded_file is not None:
        try:
            # выбираем разделитель
            if sep_mode == "auto":
                sep = None
                engine = "python"
            elif sep_mode == "tab (\\t)":
                sep = "\t"
                engine = "c"
            elif sep_mode == "space":
                sep = r"\s+"
                engine = "python"
            elif sep_mode == "custom":
                sep = custom_sep if custom_sep else None
                engine = "python" if sep is None else "c"
            else:
                sep = sep_mode  # ",", ";", "|"
                engine = "c"

            df_raw = pd.read_csv(
                uploaded_file,
                sep=sep,
                engine=engine,
                skiprows=skip_rows_upload,
                on_bad_lines="skip",
            )

            if save_uploaded:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                save_path = DATA_DIR / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                st.sidebar.success(f"Saved to /data as {uploaded_file.name}")

        except ParserError:
            st.sidebar.warning(
                "Cannot parse this CSV with current settings.\n"
                "Try another delimiter / skip rows / clean the file."
            )
        except Exception as e:
            st.sidebar.warning(f"Error reading CSV: {e}")

# ---- Загрузка из /data ----
else:
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        st.sidebar.warning("No CSV files in /data.")
    else:
        file_labels = [f.name for f in csv_files]
        selected_name = st.sidebar.selectbox(
            "Choose file from /data",
            options=file_labels,
        )
        skip_rows = st.sidebar.number_input(
            "Skip first N rows (headers/notes)",
            min_value=0,
            max_value=100,
            value=0,
            step=0,
        )
        selected_path = DATA_DIR / selected_name

        try:
            # load_data уже читает sep=None, engine='python', on_bad_lines='skip'
            df_raw = load_data(path=selected_path, skip_rows=skip_rows)
        except Exception as e:
            st.sidebar.warning(f"Error reading file from /data: {e}")

if df_raw is None:
    st.info("Load a CSV file to start analysis.")
    st.stop()

# ---- Опция транспонирования (горизонтальные/вертикальные таблицы) ----
transpose_flag = st.sidebar.checkbox(
    "Transpose table (swap rows/columns)",
    value=False,
)

if transpose_flag:
    df_raw = df_raw.T.reset_index().rename(columns={"index": "index"})

st.write("### Raw data (first rows)")
st.dataframe(df_raw.head(), use_container_width=True)
# ---- режим временного ряда ----
time_axis_mode = st.sidebar.radio(
    "Time axis mode for time series/forecast",
    ["Use date column", "Use year columns (wide)"],
)

wide_id_col = None
wide_id_value = None

if time_axis_mode == "Use year columns (wide)":
    year_cols_detected = [
        c for c in df_raw.columns
        if str(c).strip().isdigit() and len(str(c).strip()) == 4
    ]
    non_year_cols = [c for c in df_raw.columns if c not in year_cols_detected]

    if not year_cols_detected:
        st.sidebar.warning(
            "No year-like columns (1960, 1961, ...) found; "
            "switch to 'Use date column' mode."
        )
    else:
        wide_id_col = st.sidebar.selectbox(
            "ID column (country/region/indicator)",
            options=non_year_cols,
        )
        wide_id_value = st.sidebar.selectbox(
            "Row for time series",
            options=df_raw[wide_id_col].dropna().unique(),
        )


# ---------------- Выбор колонок ----------------

all_cols = df_raw.columns.tolist()
if not all_cols:
    st.warning("Dataset has no columns.")
    st.stop()

date_col = st.sidebar.selectbox(
    "Date column",
    options=all_cols,
)

target_col = st.sidebar.selectbox(
    "Target column (for forecast / regression)",
    options=all_cols,
)

# ---------------- Приведение типов ----------------

df = df_raw.copy()

# дата
df[date_col] = pd.to_datetime(
    df[date_col],
    errors="coerce",
    dayfirst=True,
)

# остальные -> пробуем сделать числовыми
for col in df.columns:
    if col == date_col:
        continue
    df[col] = (
        df[col]
        .replace("–", None)
        .replace("-", None)
        .replace(",", ".", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

numeric_cols = df.select_dtypes(include="number").columns.tolist()

# диапазон дат (глобальный фильтр для всех графиков/моделей)
if df[date_col].notna().sum() > 1:
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    date_range = st.sidebar.slider(
        "Date range for analysis",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    )
    mask = (df[date_col] >= date_range[0]) & (df[date_col] <= date_range[1])
    df = df.loc[mask].copy()
else:
    st.sidebar.warning("Date column has too few valid values.")

# дополнительные признаки
df = add_date_features(df, date_col=date_col)

# лог-трансформация и binning только если таргет числовой
if target_col in numeric_cols:
    df = add_log_transform(df, cols=[target_col])
    df = add_binning(df, col=target_col, bins=3, labels=["low", "medium", "high"])
else:
    st.sidebar.info("Target is not numeric → log/binning skipped.")

st.write("### Transformed data sample")
st.dataframe(df.head(), use_container_width=True)

# ---------------- Табы ----------------

tabs = st.tabs(
    [
        "Descriptive analysis",
        "Transformations & correlations",
        "Time series & decomposition",
        "Regression model",
        "Forecast (ARIMA)",
    ]
)

# ---------- Tab 1: Descriptive analysis ----------

with tabs[0]:
    st.subheader("Descriptive statistics")

    if not numeric_cols:
        st.warning(
            "No numeric columns found. "
            "Try another delimiter / increase 'Skip first N rows' / transpose the table."
        )
    else:
        # выбрасываем колонки, где все значения NaN
        valid_numeric_cols = [c for c in numeric_cols if df[c].count() > 0]

        if not valid_numeric_cols:
            st.warning(
                "Numeric columns exist, but all values are empty/NaN.\n"
                "Most likely you see only metadata (top rows of the file).\n"
                "Try increasing 'Skip first N rows' in the sidebar."
            )
        else:
            col_for_stats = st.selectbox(
                "Numeric column for histogram/boxplot",
                options=valid_numeric_cols,
                index=0,
            )

            stats = descriptive_stats(df, valid_numeric_cols)
            st.dataframe(stats, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_hist(df, col_for_stats), use_container_width=True)
            with col2:
                st.pyplot(plot_box(df, col_for_stats), use_container_width=True)


# ---------- Tab 2: Transformations & correlations ----------

with tabs[1]:
    st.subheader("Correlation and covariance")

    if len(numeric_cols) >= 2:
        corr = correlation_matrix(df, numeric_cols)
        cov = covariance_matrix(df, numeric_cols)

        st.write("**Correlation matrix**")
        st.dataframe(corr, use_container_width=True)

        st.write("**Covariance matrix**")
        st.dataframe(cov, use_container_width=True)

        st.pyplot(plot_heatmap(corr), use_container_width=True)
    else:
        st.warning(
            "Not enough numeric columns for correlation/covariance.\n"
            "Choose another CSV or adjust skipped header rows."
        )

# ---------- Tab 3: Time series & decomposition ----------

with tabs[2]:
    st.subheader("Time series graph")

    # ---- режим: годы в колонках (World Bank и т.п.) ----
    if time_axis_mode == "Use year columns (wide)":
        if wide_id_col is None:
            st.warning("Select ID column and row for time series in the sidebar.")
        else:
            ts_df, year_cols = build_wide_time_series(
                df_raw, wide_id_col, wide_id_value
            )
            if ts_df is None or ts_df.empty:
                st.warning(
                    "No numeric values found for this row/year columns.\n"
                    "Try another row or check 'Skip first N rows' / delimiter."
                )
            else:
                # График ряда
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(ts_df["date"], ts_df["value"])
                ax.set_title(f"Time series for {wide_id_value}")
                ax.set_xlabel("Year")
                ax.set_ylabel("Value")
                fig.autofmt_xdate()
                st.pyplot(fig, use_container_width=True)

                # Декомпозиция
                st.subheader("Decomposition (trend / seasonality / residual)")
                from statsmodels.tsa.seasonal import seasonal_decompose

                ts = ts_df.set_index("date")["value"].asfreq("YS").interpolate()

                if ts.size < 5:
                    st.warning("Too few points for decomposition (need at least ~5–6 years).")
                else:
                    period = max(2, min(10, ts.size // 3))
                    result = seasonal_decompose(ts, model="additive", period=period)
                    fig2 = result.plot()
                    fig2.set_size_inches(6, 5)
                    st.pyplot(fig2, use_container_width=True)

    # ---- обычный режим: дата в колонке ----
    else:
        if date_col not in df.columns or target_col not in numeric_cols:
            st.warning(
                "For time series select a **date column** and numeric **target column**."
            )
        else:
            ts_fig = plot_time_series(df, date_col=date_col, target_col=target_col)
            if ts_fig is not None:
                st.pyplot(ts_fig, use_container_width=True)
            else:
                st.warning("Not enough data points for time series.")

            st.subheader("Decomposition (trend / seasonality / residual)")
            dec_fig = plot_decomposition(
                df,
                date_col=date_col,
                target_col=target_col,
                freq=12,
            )
            if dec_fig is not None:
                st.pyplot(dec_fig, use_container_width=True)
            else:
                st.warning("Not enough data for decomposition.")

# ---------- Tab 4: Regression model ----------

with tabs[4]:
    st.subheader("ARIMA forecast")

    # ---- wide режим (годы в колонках) ----
    if time_axis_mode == "Use year columns (wide)":
        if wide_id_col is None:
            st.warning("Select ID column and row for time series in the sidebar.")
        else:
            ts_df, year_cols = build_wide_time_series(
                df_raw, wide_id_col, wide_id_value
            )
            if ts_df is None or ts_df.empty:
                st.warning(
                    "No numeric values found for this row/year columns.\n"
                    "Try another row or adjust CSV reading settings."
                )
            else:
                from statsmodels.tsa.arima.model import ARIMA

                ts = ts_df.set_index("date")["value"].asfreq("YS").interpolate()

                if ts.size < 5:
                    st.warning("Too few points for ARIMA (need >= 5 years).")
                else:
                    try:
                        order = (1, 1, 1)
                        model = ARIMA(ts, order=order).fit()

                        steps = st.slider(
                            "Forecast horizon (years)",
                            min_value=1,
                            max_value=20,
                            value=10,
                        )
                        forecast = model.get_forecast(steps=steps).predicted_mean

                        st.write("### Forecast values")
                        st.dataframe(
                            forecast.to_frame(name="forecast"),
                            use_container_width=True,
                        )

                        fig, ax = plt.subplots(figsize=(6, 3))
                        ts.plot(ax=ax, label="history")
                        forecast.plot(ax=ax, label="forecast")
                        ax.set_title(f"ARIMA forecast for {wide_id_value}")
                        ax.legend()
                        st.pyplot(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Unable to build ARIMA model: {e}")

    # ---- обычный режим: дата в колонке ----
    else:
        if target_col not in numeric_cols:
            st.warning("Target column must be numeric for ARIMA forecast.")
        else:
            try:
                from src.forecast import prepare_time_series, train_arima, forecast_future

                ts = prepare_time_series(
                    df,
                    date_col=date_col,
                    target_col=target_col,
                )

                if ts.dropna().shape[0] < 10:
                    st.warning("Too few points for ARIMA. Select another range/column.")
                else:
                    order = (1, 1, 1)
                    model = train_arima(ts, order=order)

                    steps = st.slider(
                        "Forecast horizon (months)",
                        min_value=3,
                        max_value=36,
                        value=12,
                    )
                    future = forecast_future(model, steps=steps)

                    st.write("### Forecast values")
                    st.dataframe(
                        future.to_frame(name="forecast"),
                        use_container_width=True,
                    )

                    fig, ax = plt.subplots(figsize=(6, 3))
                    ts.plot(ax=ax, label="history")
                    future.plot(ax=ax, label="forecast")
                    ax.set_title("ARIMA forecast")
                    ax.legend()
                    st.pyplot(fig, use_container_width=True)
            except Exception:
                st.warning(
                    "Unable to build ARIMA model for this selection.\n"
                    "Try another target column or date range."
                )


# ---------- Tab 5: Forecast (ARIMA) ----------

with tabs[4]:
    st.subheader("ARIMA forecast")

    if target_col not in numeric_cols:
        st.warning("Target column must be numeric for ARIMA forecast.")
    else:
        try:
            ts = prepare_time_series(
                df,
                date_col=date_col,
                target_col=target_col,
            )

            if ts.dropna().shape[0] < 10:
                st.warning("Too few points for ARIMA. Select another range/column.")
            else:
                order = (1, 1, 1)
                model = train_arima(ts, order=order)

                steps = st.slider(
                    "Forecast horizon (months)",
                    min_value=3,
                    max_value=36,
                    value=12,
                )
                future = forecast_future(model, steps=steps)

                st.write("### Forecast values")
                st.dataframe(
                    future.to_frame(name="forecast"),
                    use_container_width=True,
                )

                fig, ax = plt.subplots(figsize=(6, 3))
                ts.plot(ax=ax, label="history")
                future.plot(ax=ax, label="forecast")
                ax.set_title("ARIMA forecast")
                ax.legend()
                st.pyplot(fig, use_container_width=True)
        except Exception:
            st.warning(
                "Unable to build ARIMA model for this selection.\n"
                "Try another target column or date range."
            )
