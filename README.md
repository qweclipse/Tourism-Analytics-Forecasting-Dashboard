Tourism Analytics & Forecasting Dashboard

Interactive data analysis and forecasting application built with Python and Streamlit.

This project implements the full CRISP-DM workflow and provides tools for data loading, cleaning, transformation, visualization, regression modeling, and time-series forecasting. It supports both standard (long-format) datasets and wide-format datasets such as World Bank indicators.

Features
Data Loading

Import datasets from the local /data directory or via file upload.

Automatic and manual delimiter selection (auto detection, comma, semicolon, tab, space, or custom).

Ability to skip metadata rows.

Robust handling of corrupted CSV files using on_bad_lines="skip".

Support for long-format datasets (date in a column) and wide-format datasets (years as columns).

Option to save uploaded datasets into /data.

Descriptive Analytics

Summary statistics for all numeric fields.

Correlation and covariance matrices.

Histograms, box plots, and heatmaps.

Interactive date range filtering.

Data Transformations

Automatic datetime parsing.

Numeric cleaning and normalization.

Feature engineering:

Year and month extraction

Cyclic transformations (month_sin, month_cos)

Logarithmic transformation

Quantile-based binning

Conversion of wide-format datasets into time series.

Machine Learning Models
Regression (Random Forest)

Selectable feature set.

Preprocessing pipeline with support for categorical encoding.

Evaluation metrics: MAE, R².

Prediction interface for custom user input.

Time-Series Forecasting (ARIMA)

Forecasting support for both long-format and wide-format time series.

Configurable forecast horizon.

Trend, seasonal, and residual decomposition using seasonal_decompose.

Streamlit Application

Modular user interface organized into tabs.

Interactive charts and tables.

Row selection for wide datasets.

Real-time visualization of model results.


Project Structure

project/
│
├── app/
│   └── app.py                # Main Streamlit application
│
├── src/
│   ├── data_loader.py        # CSV ingestion logic
│   ├── preprocessing.py       # Cleaning and feature engineering
│   ├── eda.py                # Descriptive analytics and plots
│   ├── models.py             # Regression model
│   ├── forecast.py           # ARIMA forecasting
│   └── config.py
│
├── data/                     # Local datasets
│   └── *.csv
│
└── README.md



## Installation and Usage

Clone the repository:
git clone https://github.com/qweclipse/tourism-analytics.git
cd tourism-analytics

Create a virtual environment:
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

Install dependencies:
pip install -r requirements.txt

Run the application:
streamlit run app/app.py

The application will open in a web browser.

## Use Cases
- Analysis of tourism arrival trends.
- Forecasting country-level indicators using wide-format datasets.
- Seasonality and time-series decomposition.
- Exploratory data analysis for inconsistent or messy CSV files.
- Educational and portfolio-grade data analytics projects.

## CRISP-DM Compliance
The project follows the CRISP-DM methodology:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

## Technologies Used
- Python
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- statsmodels (ARIMA)

## Author
Sorocatii Ghennadii
Technical University of Moldova
GitHub: https://github.com/qweclipse
