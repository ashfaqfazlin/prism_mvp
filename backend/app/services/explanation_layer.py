"""Explanation layer: plain-language, directional reasoning, uncertainty, counterfactual preview."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from app.services.data_service import data_service, FEATURE_COLS, NUMERIC

# UCI Credit (anonymized) — human-readable labels for plain-language explanations
FEATURE_LABELS = {
    "A1": "checking account status",
    "A2": "loan duration",
    "A3": "credit history",
    "A4": "loan purpose",
    "A5": "credit amount",
    "A6": "savings account",
    "A7": "employment duration",
    "A8": "installment rate",
    "A9": "personal status",
    "A10": "other parties",
    "A11": "residence duration",
    "A12": "property type",
    "A13": "age",
    "A14": "other payment plans",
    "A15": "housing type",
}

# Units for numeric features
FEATURE_UNITS = {
    "A2": "months",
    "A5": "",
    "A8": "% of income",
    "A11": "years",
    "A13": "years",
    "A15": "",
}


def _enc_to_orig(enc_name: str) -> str | None:
    """Map encoded feature name (e.g. num__A2, cat__A5_0, cat__A13_0.0) to original feature (A2, A5, A13)."""
    if enc_name.startswith("num__"):
        return enc_name.replace("num__", "").strip()
    m = re.match(r"cat__([A-Z][A-Z0-9]*)_", enc_name)
    return m.group(1) if m else None


def _label(f: str) -> str:
    return FEATURE_LABELS.get(f, f)


def _format_value(feature: str, value: Any) -> str:
    """Format the value with appropriate units."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    if feature in NUMERIC:
        try:
            v = float(value)
            unit = FEATURE_UNITS.get(feature, "")
            if feature == "A5":  # credit amount
                return f"${v:,.0f}"
            elif unit:
                return f"{v:.0f} {unit}"
            else:
                return f"{v:.0f}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def plain_language_explanations(
    shap_result: dict[str, Any],
    decision: str,
    row: dict[str, Any],
    top_k: int = 5,
) -> dict[str, Any]:
    """Decision-factor contribution summaries, directional reasoning, plain-language bullets.
    
    SHAP values are for P(+) = approval probability:
    - Positive SHAP → pushed toward approval
    - Negative SHAP → pushed toward rejection
    """
    names = shap_result.get("decision_factor_names") or shap_result.get("feature_names") or []
    values = shap_result.get("values") or []
    bullets: list[str] = []
    summary: list[dict[str, Any]] = []
    
    is_approved = decision == "+"
    
    # Separate factors pushing toward approval vs rejection
    approval_factors = []
    rejection_factors = []
    
    for n, v in zip(names[:top_k * 2], values[:top_k * 2]):  # Get more to filter
        orig = _enc_to_orig(n)
        label = _label(orig) if orig else n
        raw_value = row.get(orig) if orig else row.get(n)
        formatted = _format_value(orig or n, raw_value)
        
        factor_info = {
            "feature": orig or n,
            "label": label,
            "shap_value": float(v),
            "raw_value": raw_value,
            "formatted_value": formatted,
        }
        
        if v > 0:
            approval_factors.append(factor_info)
        else:
            rejection_factors.append(factor_info)
    
    # Generate contextual directional reasoning
    if is_approved:
        if approval_factors:
            top_positive = approval_factors[0]["label"]
            directional = f"The decision engine approved this application. Key factors like {top_positive} contributed positively to this outcome."
        else:
            directional = "The decision engine approved this application based on the overall profile."
    else:
        if rejection_factors:
            top_negative = rejection_factors[0]["label"]
            directional = f"The decision engine rejected this application. Factors like {top_negative} weighed against approval."
        else:
            directional = "The decision engine rejected this application based on the overall risk assessment."
    
    # Generate intelligent bullets based on actual impact
    seen_features = set()
    
    if is_approved:
        # For approvals: show what helped (positive SHAP) and what almost hurt (negative SHAP)
        for f in approval_factors[:3]:
            if f["feature"] in seen_features:
                continue
            seen_features.add(f["feature"])
            if f["formatted_value"] != "unknown":
                bullets.append(f"Your {f['label']} ({f['formatted_value']}) positively influenced the approval.")
            else:
                bullets.append(f"Your {f['label']} positively influenced the approval.")
            summary.append({"decision_factor": f["feature"], "label": f["label"], "direction": "positive", "value": f["shap_value"]})
        
        for f in rejection_factors[:2]:
            if f["feature"] in seen_features:
                continue
            seen_features.add(f["feature"])
            if f["formatted_value"] != "unknown":
                bullets.append(f"Your {f['label']} ({f['formatted_value']}) slightly reduced approval confidence.")
            else:
                bullets.append(f"Your {f['label']} slightly reduced approval confidence.")
            summary.append({"decision_factor": f["feature"], "label": f["label"], "direction": "negative", "value": f["shap_value"]})
    else:
        # For rejections: show what hurt (negative SHAP) and what helped (positive SHAP)
        for f in rejection_factors[:3]:
            if f["feature"] in seen_features:
                continue
            seen_features.add(f["feature"])
            impact = abs(f["shap_value"])
            strength = "significantly" if impact > 0.5 else "moderately" if impact > 0.2 else ""
            if f["formatted_value"] != "unknown":
                bullets.append(f"Your {f['label']} ({f['formatted_value']}) {strength} contributed to the rejection.".replace("  ", " "))
            else:
                bullets.append(f"Your {f['label']} {strength} contributed to the rejection.".replace("  ", " "))
            summary.append({"decision_factor": f["feature"], "label": f["label"], "direction": "negative", "value": f["shap_value"]})
        
        for f in approval_factors[:2]:
            if f["feature"] in seen_features:
                continue
            seen_features.add(f["feature"])
            if f["formatted_value"] != "unknown":
                bullets.append(f"Your {f['label']} ({f['formatted_value']}) was favorable but not enough to change the outcome.")
            else:
                bullets.append(f"Your {f['label']} was favorable but not enough to change the outcome.")
            summary.append({"decision_factor": f["feature"], "label": f["label"], "direction": "positive", "value": f["shap_value"]})

    return {
        "bullets": bullets[:top_k],
        "summary": summary[:top_k],
        "directional_reasoning": directional,
    }


def uncertainty_stability(
    row_raw: dict[str, Any],
    decision: str,
    confidence: float,
    delta_pct: float = 0.05,
) -> dict[str, Any]:
    """PRISM: Confidence band, volatility check, low-confidence warning."""
    from app.services.model_service import model_service

    data_service._ensure_loaded()
    preproc = data_service.preproc
    low = confidence < 0.6
    band = "low" if confidence < 0.6 else ("medium" if confidence < 0.85 else "high")
    is_approved = decision == "+"

    # Simple stability: perturb numeric features slightly, re-predict
    stable = True
    unstable_features = []
    try:
        df = pd.DataFrame([{k: row_raw.get(k) for k in FEATURE_COLS}])
        df = df.replace("?", np.nan)
        for c in NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        base_pred = decision
        for c in NUMERIC:
            val = df[c].iloc[0]
            if pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            for sign in (-1, 1):
                df2 = df.copy()
                delta = max(1, abs(v) * delta_pct) * sign
                df2[c] = v + delta
                X2 = preproc.transform(df2)
                labels, _ = model_service.predict(X2)
                p2 = "+" if int(labels[0]) == 1 else "-"
                if p2 != base_pred:
                    stable = False
                    unstable_features.append(_label(c))
                    break
    except Exception:
        pass

    # Generate contextual warnings
    warning = None
    volatility_note = None
    
    if low and not stable:
        if is_approved:
            warning = f"This approval has low confidence ({confidence*100:.0f}%) and is sensitive to small changes. The decision could easily flip to rejection."
        else:
            warning = f"This rejection has low confidence ({confidence*100:.0f}%) and is borderline. Small improvements could lead to approval."
    elif low:
        if is_approved:
            warning = f"This approval has low confidence ({confidence*100:.0f}%). Consider it a tentative positive assessment."
        else:
            warning = f"This rejection has low confidence ({confidence*100:.0f}%). The application is borderline — small changes might affect the outcome."
    elif not stable:
        if unstable_features:
            features_str = ", ".join(unstable_features[:2])
            if is_approved:
                warning = f"This approval is sensitive to changes in {features_str}. Small negative changes could flip the decision."
            else:
                warning = f"This rejection is sensitive to changes in {features_str}. Small improvements could lead to approval."
        else:
            warning = "This decision is unstable under small input changes."
    
    if not stable:
        if is_approved:
            volatility_note = "The decision engine indicates this approval is near the decision boundary. Monitor for changes in the applicant's financial situation."
        else:
            volatility_note = "The decision engine indicates this case is near the decision boundary. Use the What-if tool to explore potential improvements."

    return {
        "confidence_band": band,
        "stable": stable,
        "warning": warning,
        "volatility_note": volatility_note,
    }


def counterfactual_preview(
    row: dict[str, Any],
    counterfactuals: list[dict[str, Any]],
    shap_result: dict[str, Any],
    decision: str,
) -> list[dict[str, Any]]:
    """PRISM: Minimal-change suggestions that could flip the decision. From DiCE or SHAP fallback."""
    out: list[dict[str, Any]] = []
    names = shap_result.get("decision_factor_names") or shap_result.get("feature_names") or []
    values = shap_result.get("values") or []
    is_rejected = decision == "-"

    # 1. From DiCE counterfactuals (most accurate)
    for cf in (counterfactuals or []):
        ch = cf.get("changed_decision_factors") or cf.get("changed_features") or []
        mod = cf.get("modified") or {}
        orig = cf.get("original") or {}
        for f in ch:
            o, m = orig.get(f), mod.get(f)
            if o is None and m is None:
                continue
            label = _label(f)
            if f in NUMERIC and isinstance(o, (int, float)) and isinstance(m, (int, float)):
                try:
                    diff = float(m) - float(o)
                    orig_fmt = _format_value(f, o)
                    new_fmt = _format_value(f, m)
                    if is_rejected:
                        out.append({
                            "suggestion": f"If your {label} were {new_fmt} instead of {orig_fmt}, approval would be more likely.",
                            "decision_factor": f,
                            "change": diff,
                            "from_value": o,
                            "to_value": m,
                        })
                    else:
                        direction = "higher" if diff > 0 else "lower"
                        out.append({
                            "suggestion": f"If your {label} were {direction} ({new_fmt}), the outcome might change.",
                            "decision_factor": f,
                            "change": diff,
                            "from_value": o,
                            "to_value": m,
                        })
                except (TypeError, ValueError):
                    out.append({"suggestion": f"Changing your {label} could affect the outcome.", "decision_factor": f, "change": None})
            else:
                out.append({"suggestion": f"A different {label} could affect the outcome.", "decision_factor": f, "change": None})

    # 2. SHAP-based fallback: find factors that hurt the desired outcome
    if not out and values:
        # For rejections: factors with negative SHAP pushed toward rejection
        # Improving those could flip to approval
        for n, v in zip(names, values):
            if len(out) >= 3:
                break
            orig_feature = _enc_to_orig(n)
            if not orig_feature:
                continue
            label = _label(orig_feature)
            raw_value = row.get(orig_feature)
            formatted = _format_value(orig_feature, raw_value)
            
            if is_rejected and v < 0:
                # This factor pushed toward rejection
                if orig_feature in NUMERIC and raw_value is not None:
                    try:
                        current = float(raw_value)
                        # Suggest improvement direction based on common credit logic
                        if orig_feature == "A2":  # loan duration
                            out.append({"suggestion": f"A shorter loan duration (currently {formatted}) could improve approval chances.", "decision_factor": orig_feature, "change": None})
                        elif orig_feature == "A5":  # credit amount
                            out.append({"suggestion": f"A lower credit amount (currently {formatted}) might result in approval.", "decision_factor": orig_feature, "change": None})
                        elif orig_feature == "A8":  # installment rate
                            out.append({"suggestion": f"A lower installment rate (currently {formatted}) could help with approval.", "decision_factor": orig_feature, "change": None})
                        elif orig_feature == "A11":  # residence duration
                            out.append({"suggestion": f"Longer residence duration (currently {formatted}) could strengthen your application.", "decision_factor": orig_feature, "change": None})
                        else:
                            out.append({"suggestion": f"Improving your {label} (currently {formatted}) could positively impact the decision.", "decision_factor": orig_feature, "change": None})
                    except (TypeError, ValueError):
                        out.append({"suggestion": f"Improving your {label} could help achieve approval.", "decision_factor": orig_feature, "change": None})
                else:
                    out.append({"suggestion": f"A stronger {label} could improve your chances.", "decision_factor": orig_feature, "change": None})
            elif not is_rejected and v > 0:
                # For approvals: show what could threaten it
                out.append({"suggestion": f"Your {label} ({formatted}) is favorable — changes here could affect the outcome.", "decision_factor": orig_feature, "change": None})

    # If still no suggestions for rejected applications, provide general guidance
    if not out and is_rejected:
        out.append({"suggestion": "Consider improving your credit history or reducing the loan amount to increase approval chances.", "decision_factor": "general", "change": None})

    return out[:5]


def get_feature_ranges() -> dict[str, dict[str, float]]:
    """PRISM: Min/max for numeric decision factors (sliders). From default dataset."""
    try:
        df = data_service.load_default_dataset()
    except FileNotFoundError:
        return {}
    ranges: dict[str, dict[str, float]] = {}
    for c in NUMERIC:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.size:
            ranges[c] = {"min": float(s.min()), "max": float(s.max()), "label": FEATURE_LABELS.get(c, c)}
    return ranges
