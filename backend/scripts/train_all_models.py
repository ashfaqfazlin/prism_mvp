"""Train models for all registered domains.

Usage:
    python scripts/train_all_models.py              # Train all domains
    python scripts/train_all_models.py german_credit  # Train specific domain
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Add backend root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.domain_config import DOMAIN_REGISTRY, DomainConfig, is_model_trained

DATASETS_DIR = ROOT / "datasets"


def load_dataset(domain: DomainConfig) -> pd.DataFrame:
    """Load dataset for a domain."""
    # Map domain ID to dataset filename
    filename_map = {
        "uci_credit_approval": "uci_credit_approval.csv",
        "german_credit": "german_credit.csv",
        "taiwan_credit_card": "taiwan_credit_card.csv",
    }
    
    filename = filename_map.get(domain.id)
    if not filename:
        raise ValueError(f"No dataset file mapped for domain: {domain.id}")
    
    filepath = DATASETS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} rows from {filename}")
    return df


def prepare_data(df: pd.DataFrame, domain: DomainConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for training."""
    df = df.copy()
    
    # Handle missing values marker
    df = df.replace("?", np.nan)
    
    # Convert numeric columns
    for col in domain.numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Get target column
    target_col = domain.target_col
    if target_col not in df.columns:
        # Try alternate names
        alt_names = ["class", "target", "label", "y"]
        for alt in alt_names:
            if alt in df.columns:
                target_col = alt
                break
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{domain.target_col}' not found. Available: {df.columns.tolist()}")
    
    # Map target values to binary
    if domain.positive_value == "+" and domain.negative_value == "-":
        df["_target"] = df[target_col].map({"+": 1, "-": 0})
    elif domain.positive_value == 1 and domain.negative_value == 2:
        # German credit: 1=good, 2=bad -> we predict "bad" as positive for default prediction
        df["_target"] = df[target_col].map({1: 0, 2: 1})  # 1=good->0, 2=bad->1
    elif domain.positive_value == 1 and domain.negative_value == 0:
        df["_target"] = df[target_col].astype(int)
    else:
        # Generic binary mapping
        df["_target"] = (df[target_col] == domain.positive_value).astype(int)
    
    # Drop rows with missing target
    df = df.dropna(subset=["_target"])
    
    # Filter to available feature columns
    available_features = [c for c in domain.feature_cols if c in df.columns]
    if len(available_features) < len(domain.feature_cols):
        missing = set(domain.feature_cols) - set(available_features)
        print(f"  Warning: Missing features: {missing}")
    
    X = df[available_features].copy()
    y = df["_target"].astype(int)
    
    return X, y


def build_preprocessing(X: pd.DataFrame, domain: DomainConfig) -> ColumnTransformer:
    """Build preprocessing pipeline for a domain."""
    # Get actual categorical and numeric columns that exist
    cat_cols = [c for c in domain.categorical_cols if c in X.columns]
    num_cols = [c for c in domain.numeric_cols if c in X.columns]
    
    # Imputers
    impute_cat = SimpleImputer(strategy="most_frequent")
    impute_num = SimpleImputer(strategy="median")
    
    # Encoders/scalers
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()
    
    # Pipelines
    transformers = []
    if cat_cols:
        cat_pipe = Pipeline([("impute", impute_cat), ("ohe", ohe)])
        transformers.append(("cat", cat_pipe, cat_cols))
    if num_cols:
        num_pipe = Pipeline([("impute", impute_num), ("scale", scaler)])
        transformers.append(("num", num_pipe, num_cols))
    
    preproc = ColumnTransformer(transformers, remainder="drop")
    return preproc


def get_feature_names(preproc: ColumnTransformer, domain: DomainConfig) -> list[str]:
    """Extract feature names after preprocessing."""
    if hasattr(preproc, "get_feature_names_out"):
        return list(preproc.get_feature_names_out())
    
    # Manual extraction
    names = []
    for name, pipe, cols in preproc.transformers_:
        if name == "cat":
            ohe = pipe.named_steps["ohe"]
            for i, c in enumerate(cols):
                for cat in ohe.categories_[i]:
                    names.append(f"{c}_{cat}")
        elif name == "num":
            names.extend(cols)
    return names


def train_domain(domain: DomainConfig, force: bool = False) -> dict:
    """Train model for a specific domain."""
    print(f"\n{'='*60}")
    print(f"Training: {domain.name} ({domain.id})")
    print(f"{'='*60}")
    
    # Check if already trained
    if not force and is_model_trained(domain.id):
        print(f"  Model already exists. Use --force to retrain.")
        return {"status": "skipped", "reason": "already_trained"}
    
    # Create artifacts directory
    domain.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    try:
        df = load_dataset(domain)
        X, y = prepare_data(df, domain)
        print(f"  Features: {X.shape[1]}, Samples: {len(X)}")
        print(f"  Target distribution: {y.value_counts().to_dict()}")
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        return {"status": "error", "reason": str(e)}
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build and fit preprocessing
    print("  Building preprocessing pipeline...")
    preproc = build_preprocessing(X_train, domain)
    X_train_t = preproc.fit_transform(X_train)
    X_test_t = preproc.transform(X_test)
    
    feature_names = get_feature_names(preproc, domain)
    print(f"  Encoded features: {len(feature_names)}")
    
    # Train XGBoost
    print("  Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_train_t, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_t)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, 
                                target_names=[domain.negative_label, domain.positive_label]))
    
    # Save artifacts
    meta = {
        "domain_id": domain.id,
        "domain_name": domain.name,
        "feature_cols": list(X.columns),
        "target_col": domain.target_col,
        "categorical": [c for c in domain.categorical_cols if c in X.columns],
        "numeric": [c for c in domain.numeric_cols if c in X.columns],
        "encoded_feature_names": feature_names,
        "positive_label": domain.positive_label,
        "negative_label": domain.negative_label,
        "feature_labels": domain.feature_labels,
        "test_accuracy": accuracy,
    }
    
    joblib.dump({"preproc": preproc, "meta": meta}, domain.preprocessing_path)
    joblib.dump(model, domain.model_path)
    
    # Save meta as JSON for inspection
    with open(domain.artifacts_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"  Saved artifacts to {domain.artifacts_dir}")
    
    return {"status": "success", "accuracy": accuracy}


def main():
    """Train models for all or specified domains."""
    # Parse args
    force = "--force" in sys.argv
    domain_ids = [a for a in sys.argv[1:] if not a.startswith("--")]
    
    if domain_ids:
        domains_to_train = [DOMAIN_REGISTRY[d] for d in domain_ids if d in DOMAIN_REGISTRY]
    else:
        domains_to_train = list(DOMAIN_REGISTRY.values())
    
    print(f"Training {len(domains_to_train)} domain(s)...")
    
    results = {}
    for domain in domains_to_train:
        result = train_domain(domain, force=force)
        results[domain.id] = result
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for domain_id, result in results.items():
        status = result.get("status", "unknown")
        if status == "success":
            acc = result.get("accuracy", 0)
            print(f"  ✓ {domain_id}: {acc:.2%} accuracy")
        elif status == "skipped":
            print(f"  ○ {domain_id}: skipped (already trained)")
        else:
            print(f"  ✗ {domain_id}: {result.get('reason', 'error')}")


if __name__ == "__main__":
    main()
