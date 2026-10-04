"""Load and clean the Kaggle Credit Risk Dataset (credit_risk_dataset.csv)."""
from pathlib import Path
import pandas as pd
from .features import CATEGORICAL, NUMERIC, TARGET

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "credit_risk_dataset.csv"


def load(path=None):
    raw = pd.read_csv(path or DEFAULT_PATH)
    report = {"raw_rows": len(raw)}
    df = raw.drop_duplicates()
    report["duplicates_removed"] = len(raw) - len(df)
    n = len(df)
    # impossible values present in the raw file (e.g. age 144, 123 years of employment)
    df = df[(df.person_age <= 80) & (df.person_emp_length.fillna(0) <= 50)
            & (df.person_emp_length.fillna(0) <= df.person_age - 14)]
    report["outliers_removed"] = n - len(df)
    report["missing"] = {c: int(v) for c, v in df[NUMERIC].isna().sum().items() if v}
    report["clean_rows"] = len(df)
    return df[NUMERIC + list(CATEGORICAL) + [TARGET]].reset_index(drop=True), report
