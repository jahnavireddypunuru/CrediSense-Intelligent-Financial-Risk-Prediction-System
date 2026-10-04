# CrediSense: Intelligent Financial Risk Prediction System

Web app that predicts loan default risk from applicant details.
Stack: **FastAPI** + Jinja2 + SQLite + scikit-learn + Chart.js (bundled, works offline).

## Run
```
pip install -r requirements.txt
python app.py            # open http://127.0.0.1:8000  (register, then sign in)
```
Models are already trained in `models/`. To retrain: `python -m ml.train`
(takes about 2 minutes; `--data other.csv` trains on another file with the same columns).
Interactive API docs are generated at `/docs`.

## Dataset
Kaggle **Credit Risk Dataset** (`data/credit_risk_dataset.csv`, by laotse): 32,581 real loan records.
Cleaning: 165 duplicates and 9 impossible rows (e.g. age 144) removed; missing employment length and
interest rate filled with the median. 32,407 rows used, 21.9% defaulted. Target: `loan_status` (1 = default).
Note: amounts in this dataset are in US dollars, so the app shows `$`.

## Algorithms (none of the original Decision Tree / Random Forest / KNN / XGBoost)
| Model | Test ROC-AUC | Idea |
|---|---|---|
| Logistic Regression | 0.884 | L2-regularised sigmoid classifier |
| Support Vector Machine | 0.914 | RBF kernel (Nystroem approximation), calibrated probabilities |
| Gaussian Naive Bayes | 0.846 | Bayes' theorem on power-transformed features |
| Neural Network (MLP) | **0.919** | 2 hidden layers (48, 24), back-propagation, early stopping |

The best model by held-out ROC-AUC (the neural network) makes predictions. The decision threshold is tuned on
cross-validated training predictions, so the 20% test set is never used for tuning.

## Pipeline
Input -> feature engineering (loan-to-income, log income/loan, employment-to-age, rate x burden, one-hot)
-> median imputation -> scaling -> model -> default probability -> Low / Moderate / High band.
The result page explains each prediction with a what-if factor analysis and gives recommendations.

## Pages
Login, Register, Dashboard (KPIs and charts), New assessment, Result, History (search, filter,
delete, CSV export), Model lab (dataset, metrics, ROC curves, confusion matrix), dark mode.

## Notes for your report / viva
- `loan_grade` and `loan_int_rate` are assigned by the lender, so they already carry some risk information.
  This is standard for this dataset, but mention it when discussing results.
- Run behind a proper server and change `secret_key` in `app.py` for any real deployment.
