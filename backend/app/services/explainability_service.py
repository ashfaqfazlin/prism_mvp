"""SHAP and counterfactual (DiCE) explanations for PRISM."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from app.config import settings
from app.services.data_service import data_service, FEATURE_COLS, NUMERIC, TARGET_COL
from app.services.model_service import model_service


class ExplainabilityService:
    """SHAP attributions and DiCE counterfactuals."""

    def __init__(self) -> None:
        self._dice_data = None
        self._dice_model = None
        self._dice_exp = None
        self._shap_explainer = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        data_service._ensure_loaded()
        model_service._ensure_loaded()
        self._build_shap()
        try:
            self._build_dice()
        except Exception:
            self._dice_data = self._dice_model = self._dice_exp = None
        self._loaded = True

    def _build_shap(self) -> None:
        """Build TreeExplainer; use small background sample."""
        preproc = data_service.preproc
        model = model_service.model
        base = Path(settings.base_dir)
        default_path = base / settings.default_dataset_path
        if not default_path.exists():
            self._shap_explainer = None
            return
        df = pd.read_csv(default_path)
        df = df[FEATURE_COLS].replace("?", np.nan)
        for c in NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        X = preproc.transform(df.head(100))
        self._shap_explainer = shap.TreeExplainer(model, X, feature_names=data_service.feature_names)

    def _build_dice(self) -> None:
        """Build DiCE data + model (sklearn Pipeline) and explainer."""
        try:
            import dice_ml
        except ImportError:
            self._dice_data = self._dice_model = self._dice_exp = None
            return
        base = Path(settings.base_dir)
        default_path = base / settings.default_dataset_path
        if not default_path.exists():
            self._dice_data = self._dice_model = self._dice_exp = None
            return
        df = pd.read_csv(default_path)
        df = df[FEATURE_COLS + [TARGET_COL]].copy()
        df = df.replace("?", np.nan)
        for c in NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["outcome"] = (df[TARGET_COL].astype(str).str.strip() == "+").astype(int)
        df = df.drop(columns=[TARGET_COL])

        preproc = data_service.preproc
        model = model_service.model
        full_model = Pipeline([("preprocessor", preproc), ("classifier", model)])

        self._dice_data = dice_ml.Data(
            dataframe=df,
            continuous_features=NUMERIC,
            outcome_name="outcome",
        )
        self._dice_model = dice_ml.Model(model=full_model, backend="sklearn")
        self._dice_exp = dice_ml.Dice(self._dice_data, self._dice_model, method="random")

    def shap_values(self, x: np.ndarray, top_k: int = 20) -> dict[str, Any]:
        """SHAP values for one encoded instance. x: (n_features,) or (1, n_features)."""
        self._ensure_loaded()
        pred, _, _ = model_service.predict_single(np.asarray(x).reshape(-1)[:])
        default = {
            "feature_names": data_service.feature_names,
            "values": [],
            "base_value": 0.0,
            "prediction": pred,
        }
        if self._shap_explainer is None:
            return default
        try:
            x = np.asarray(x).reshape(1, -1)
            vals = self._shap_explainer.shap_values(x)
            if isinstance(vals, list):
                vals = vals[1]
            vals = np.asarray(vals)
            if vals.ndim == 2:
                vals = vals[0]
            base = self._shap_explainer.expected_value
            if isinstance(base, np.ndarray):
                base = float(base[1])
            else:
                base = float(base)
            names = data_service.feature_names
            assert len(names) == len(vals)
            # Top-k by |value| for readability
            idx = np.argsort(np.abs(vals))[::-1][:top_k]
            return {
                "feature_names": [names[i] for i in idx],
                "values": [float(vals[i]) for i in idx],
                "base_value": base,
                "prediction": pred,
            }
        except Exception:
            return default

    def counterfactuals(self, query_raw: pd.DataFrame, total_cfs: int = 2) -> list[dict[str, Any]]:
        """Generate counterfactuals for one raw row. Returns list of {original, modified, ...}."""
        self._ensure_loaded()
        if self._dice_exp is None:
            return []
        query = query_raw[FEATURE_COLS].copy()
        query = query.replace("?", np.nan)
        for c in NUMERIC:
            query[c] = pd.to_numeric(query[c], errors="coerce")
        try:
            res = self._dice_exp.generate_counterfactuals(
                query, total_CFs=total_cfs, desired_class="opposite", verbose=False
            )
        except Exception:
            return []
        out = []
        orig = query.iloc[0].to_dict()
        orig_pred, _, _ = model_service.predict_single(data_service.transform(query)[0])
        cf_list = res.cf_examples_list
        if not cf_list:
            return []
        cf_df = cf_list[0].final_cfs_df
        if cf_df is None or cf_df.empty:
            return []
        outcome_col = "outcome"
        feat_cols = [c for c in cf_df.columns if c != outcome_col]
        for _, row in cf_df.iterrows():
            mod = {c: row[c] for c in feat_cols}
            mod_df = pd.DataFrame([mod])
            mod_pred = "+" if int(row.get(outcome_col, 0)) == 1 else "-"
            changed = [c for c in feat_cols if orig.get(c) != mod.get(c)]
            out.append({
                "original": {k: _v(orig.get(k)) for k in feat_cols},
                "modified": {k: _v(mod.get(k)) for k in feat_cols},
                "original_prediction": orig_pred,
                "new_prediction": mod_pred,
                "changed_features": changed,
            })
        return out


def _v(x: Any) -> Any:
    if isinstance(x, (np.integer, np.floating)):
        return float(x) if np.issubdtype(type(x), np.floating) else int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


explainability_service = ExplainabilityService()
