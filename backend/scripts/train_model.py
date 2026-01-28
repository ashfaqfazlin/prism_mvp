"""Fetch UCI Credit Approval, preprocess, train XGBoost, save artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Add backend root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# UCI Credit Approval schema (A1–A15 features, A16 target)
FEATURE_COLS = [f"A{i}" for i in range(1, 16)]
TARGET_COL = "A16"
CATEGORICAL = ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]
NUMERIC = ["A2", "A3", "A8", "A11", "A14", "A15"]


def fetch_credit_data() -> pd.DataFrame:
    """Load UCI Credit Approval (ucimlrepo or fallback URL)."""
    try:
        from ucimlrepo import fetch_ucirepo
        credit = fetch_ucirepo(id=27)
        X = credit.data.features
        y = credit.data.targets
        if hasattr(y, "columns"):
            y = y.iloc[:, 0]
        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        df.columns = FEATURE_COLS + [TARGET_COL]
        return df
    except Exception as e:
        print(f"ucimlrepo failed: {e}. Trying direct UCI URL...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
    df = pd.read_csv(url, header=None)
    df.columns = FEATURE_COLS + [TARGET_COL]
    return df


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean and split features/target."""
    df = df.copy()
    # Replace ? with NaN
    df = df.replace("?", np.nan)
    # Numerics
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Target: + -> 1, - -> 0
    df[TARGET_COL] = df[TARGET_COL].map({"+": 1, "-": 0})
    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    X = df[FEATURE_COLS]
    return X, y


def build_preprocessing(X: pd.DataFrame) -> Pipeline:
    """Build preprocessing pipeline (impute, encode, scale)."""
    from sklearn.preprocessing import OneHotEncoder

    impute_cat = SimpleImputer(strategy="most_frequent")
    impute_num = SimpleImputer(strategy="median")
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()

    cat_pipe = Pipeline([("impute", impute_cat), ("ohe", ohe)])
    num_pipe = Pipeline([("impute", impute_num), ("scale", scaler)])

    preproc = ColumnTransformer(
        [
            ("cat", cat_pipe, CATEGORICAL),
            ("num", num_pipe, NUMERIC),
        ],
        remainder="drop",
    )
    return preproc


def get_feature_names(preproc: Pipeline) -> list[str]:
    """Extract feature names after ColumnTransformer (OHE + numeric)."""
    if hasattr(preproc, "get_feature_names_out"):
        return list(preproc.get_feature_names_out())
    ct = preproc
    names = []
    for name, pipe, cols in ct.transformers_:
        if name == "cat":
            ohe = pipe.named_steps["ohe"]
            cats = ohe.categories_
            for i, c in enumerate(cols):
                for cat in cats[i]:
                    names.append(f"{c}_{cat}")
        elif name == "num":
            names.extend(cols)
    return names


def main() -> None:
    print("Fetching UCI Credit Approval...")
    raw = fetch_credit_data()
    X, y = prepare(raw)
    print(f"Shape: {X.shape}, target distribution: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Building preprocessing...")
    preproc = build_preprocessing(X_train)
    Xt = preproc.fit_transform(X_train)
    feature_names = get_feature_names(preproc)
    print(f"Encoded features: {len(feature_names)}")

    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(Xt, y_train)

    from sklearn.metrics import accuracy_score, classification_report
    X_test_t = preproc.transform(X_test)
    pred = model.predict(X_test_t)
    print(f"Test accuracy: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred, target_names=["Rejected", "Approved"]))

    # Persist
    meta = {
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "categorical": CATEGORICAL,
        "numeric": NUMERIC,
        "encoded_feature_names": feature_names,
    }
    joblib.dump({"preproc": preproc, "meta": meta}, ARTIFACTS / "preprocessing.joblib")
    joblib.dump(model, ARTIFACTS / "model.joblib")
    raw.to_csv(ARTIFACTS / "credit_approval.csv", index=False)
    with open(ARTIFACTS / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved artifacts to", ARTIFACTS)


if __name__ == "__main__":
    main()
