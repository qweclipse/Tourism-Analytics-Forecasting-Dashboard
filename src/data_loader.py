# src/data_loader.py

from pathlib import Path
import pandas as pd

from .config import DATA_DIR


def load_data(
    path: Path | str | None = None,
    skip_rows: int = 0,
) -> pd.DataFrame:
    """
    Загружает CSV.
    Если path=None — берём первый .csv из папки data.
    Проблемные строки пропускаем (on_bad_lines='skip').
    """
    if path is None:
        csv_files = list(DATA_DIR.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in data/")
        path = csv_files[0]

    path = Path(path)

    df = pd.read_csv(
        path,
        skiprows=skip_rows,
        sep=None,           # авто-детект разделителя
        engine="python",
        on_bad_lines="skip"  # строки с другим числом колонок пропускаем
    )
    return df
