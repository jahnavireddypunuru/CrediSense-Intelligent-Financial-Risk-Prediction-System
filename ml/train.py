"""Train and compare credit-risk models.

Algorithms (all different from the original proposal of DT / RF / KNN / XGBoost):
  1. Logistic Regression   - L2-regularised linear probabilistic classifier
  2. Support Vector Machine - RBF kernel (Nystroem approximation) with calibrated probabilities
  3. Gaussian Naive Bayes   - generative model with power-transformed inputs
  4. Neural Network (MLP)   - 2 hidden layers, early stopping

Run:  python -m ml.train [--data path.csv]
"""
import argparse
import warnings
warnings.filterwarnings('ignore')
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.calibration import CalibratedClassifierCV

from .dataset import load
from .features import ENGINEERED_COLUMNS, TARGET, Engineer

OUT = Path(__file__).resolve().parent.parent / "models"


def build_models():
    def pipe(clf, scaler=None):
        return Pipeline([
            ("engineer", Engineer()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", scaler or StandardScaler()),
            ("clf", clf),
        ])
    return {
        "Logistic Regression": pipe(LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")),
        # RBF-kernel SVM, scaled to 26k rows with a Nystroem kernel approximation + calibrated probabilities
        "Support Vector Machine": pipe(CalibratedClassifierCV(Pipeline([
            ("rbf", Nystroem(kernel="rbf", gamma=0.03, n_components=400, random_state=7)),
            ("svm", LinearSVC(C=1.0, class_weight="balanced", max_iter=5000))]), cv=3, method="sigmoid")),
        "Gaussian Naive Bayes": pipe(GaussianNB(), PowerTransformer()),
        "Neural Network (MLP)": pipe(MLPClassifier(hidden_layer_sizes=(48, 24), alpha=1e-3,
                                                   early_stopping=True, max_iter=400, random_state=7)),
    }


def main(data=None):
    OUT.mkdir(exist_ok=True)
    df, report = load(data)
    print("data report:", report)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"rows={len(df)}  default-rate={y.mean():.1%}  train={len(Xtr)} test={len(Xte)}")

    cv = StratifiedKFold(5, shuffle=True, random_state=1)
    results, fitted = [], {}
    for name, model in build_models().items():
        t0 = time.time()
        cv_auc = cross_val_score(model, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1)
        # decision threshold tuned on out-of-fold training predictions (test set stays untouched)
        oof = cross_val_predict(model, Xtr, ytr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        thr_grid = np.linspace(0.15, 0.7, 56)
        thr = float(thr_grid[np.argmax([f1_score(ytr, oof >= t) for t in thr_grid])])
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= thr).astype(int)
        fpr, tpr, _ = roc_curve(yte, proba)
        idx = np.linspace(0, len(fpr) - 1, 60).astype(int)
        results.append(dict(
            name=name,
            accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred),
            recall=recall_score(yte, pred), f1=f1_score(yte, pred),
            roc_auc=roc_auc_score(yte, proba), cv_auc_mean=float(cv_auc.mean()),
            cv_auc_std=float(cv_auc.std()), threshold=thr,
            confusion=confusion_matrix(yte, pred).tolist(),
            roc=dict(fpr=fpr[idx].round(4).tolist(), tpr=tpr[idx].round(4).tolist()),
            train_seconds=round(time.time() - t0, 2)))
        fitted[name] = model
        r = results[-1]
        print(f"{name:24s} AUC={r['roc_auc']:.3f} F1={r['f1']:.3f} acc={r['accuracy']:.3f} thr={thr:.2f}")

    best = max(results, key=lambda r: (r["roc_auc"], r["f1"]))
    print("Selected model:", best["name"])

    # reference values used for the per-applicant explanation
    ref = X.median(numeric_only=True).to_dict()
    ref.update({c: X[c].mode()[0] for c in X.select_dtypes(exclude="number").columns})

    joblib.dump(fitted[best["name"]], OUT / "best_model.joblib")
    joblib.dump(fitted, OUT / "all_models.joblib")
    meta = dict(best=best["name"], threshold=best["threshold"], reference=ref,
                default_rate=float(y.mean()), rows=int(len(df)), data_report=report, results=results,
                feature_columns=ENGINEERED_COLUMNS)
    (OUT / "metrics.json").write_text(json.dumps(meta, indent=2, default=float))
    print("saved ->", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    main(ap.parse_args().data)
