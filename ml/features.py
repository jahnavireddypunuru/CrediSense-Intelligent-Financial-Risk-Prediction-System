"""Feature schema for the Kaggle "Credit Risk Dataset" (laotse) shared by training and the web app."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET = "loan_status"  # 1 = defaulted, 0 = repaid

NUMERIC = ["person_age", "person_income", "person_emp_length", "loan_amnt",
           "loan_int_rate", "cb_person_cred_hist_length"]
CATEGORICAL = {
    "person_home_ownership": ["RENT", "MORTGAGE", "OWN", "OTHER"],
    "loan_intent": ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
    "cb_person_default_on_file": ["N", "Y"],
}
DERIVED = ["loan_percent_income", "log_income", "log_loan", "emp_to_age", "rate_x_burden"]

LABELS = {
    "person_age": "Age", "person_income": "Annual income", "person_emp_length": "Employment length",
    "loan_amnt": "Loan amount", "loan_int_rate": "Interest rate", "cb_person_cred_hist_length": "Credit history length",
    "person_home_ownership": "Home ownership", "loan_intent": "Loan purpose",
    "loan_grade": "Loan grade", "cb_person_default_on_file": "Past default on file",
}
VALUE_LABELS = {
    "RENT": "Rent", "MORTGAGE": "Mortgage", "OWN": "Own", "OTHER": "Other",
    "PERSONAL": "Personal", "EDUCATION": "Education", "MEDICAL": "Medical", "VENTURE": "Venture / business",
    "HOMEIMPROVEMENT": "Home improvement", "DEBTCONSOLIDATION": "Debt consolidation",
    "N": "No", "Y": "Yes",
}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio features and one-hot encode categoricals in a fixed column order."""
    d = df.copy()
    inc = d["person_income"].clip(lower=1)
    d["loan_percent_income"] = d["loan_amnt"] / inc
    d["log_income"] = np.log1p(inc)
    d["log_loan"] = np.log1p(d["loan_amnt"])
    d["emp_to_age"] = d["person_emp_length"] / d["person_age"].clip(lower=18)
    d["rate_x_burden"] = d["loan_int_rate"] * d["loan_percent_income"]
    for col, levels in CATEGORICAL.items():
        for lv in levels:
            d[f"{col}={lv}"] = (d[col] == lv).astype(float)
    cols = NUMERIC + DERIVED + [f"{c}={lv}" for c, ls in CATEGORICAL.items() for lv in ls]
    return d[cols].astype(float)


ENGINEERED_COLUMNS = engineer(pd.DataFrame([{
    **{n: 1 for n in NUMERIC}, **{c: ls[0] for c, ls in CATEGORICAL.items()}
}])).columns.tolist()


class Engineer(BaseEstimator, TransformerMixin):
    """Raw applicant frame -> engineered numeric matrix (lives inside the pipeline)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return engineer(X)
