"""Unified XAI contract helpers used across API endpoints.

Keeps response semantics consistent for:
- local_attribution (SHAP)
- global_importance (aggregated SHAP)
- action guidance (suggested changes)
- optimization-backed counterfactuals
"""
from __future__ import annotations


def build_explanation_taxonomy(
    has_true_counterfactuals: bool,
    has_suggested_changes: bool,
) -> dict[str, str]:
    return {
        "local_attribution": "shap_local",
        "global_importance": "shap_global_mean_abs",
        "action_guidance": "suggested_changes" if has_suggested_changes else "none",
        "counterfactuals": "optimization_based" if has_true_counterfactuals else "none",
    }
