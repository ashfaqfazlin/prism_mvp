# PRISM — Human-Centred Explainable AI (XAI) System

An interactive explainable AI framework for credit-approval decision support. PRISM combines **SHAP** (feature attributions) and **counterfactual** ("what-if") explanations in a human-centred interface.

## Features

- **PRISM** — Human-centred Explainable AI. **Decision engine**, **confidence**, **decision factors**.
- **CSV upload** — Analyze tabular datasets (UCI Credit Approval or custom)
- **Decision engine** — Black-box classifier; outputs **decision** and **confidence** (no “accuracy” in user-facing UI)
- **Explanation layer** — **Decision factor** contribution summaries, directional reasoning, plain-language bullets (e.g. *“Loan duration increased approval likelihood”*)
- **Uncertainty & stability** — Confidence bands, volatility check, low-confidence / instability warnings (e.g. *“This decision is unstable under small input changes”*)
- **Counterfactual preview** — Minimal-change suggestions that could flip the outcome (e.g. *“Increasing savings by X could change the outcome”*)
- **User interaction** — Explanation mode toggle (Plain language / Technical / What-if), sliders for numeric **decision factors**, what-if scenarios, “Update decision”
- **SHAP** — Technical view: **decision factor** impact bar chart (top 20)
- **Export** — PRISM CSV/PDF reports for auditability
- **Feedback** — Log user evaluation for research

## Tech Stack

- **Backend:** Python 3.10+, FastAPI, XGBoost, SHAP, DiCE (counterfactuals)
- **Frontend:** React 18, Vite
- **Data:** UCI Credit Approval (default); custom CSV supported

## Quick Start

```bash
# 1. Backend
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/train_model.py   # fetch UCI Credit, train XGBoost, save artifacts
PYTHONPATH=. uvicorn app.main:app --reload --port 8000

# 2. Frontend (new terminal)
cd frontend
npm install
npm run dev
```

- **App:** http://localhost:5173 (Vite proxies /api → backend)  
- **API:** http://localhost:8000  
- **Docs:** http://localhost:8000/docs  

## Project Structure

```
prism_mvp/
├── backend/
│   ├── app/
│   │   ├── api/        # Routes: upload, predict, export, feedback
│   │   ├── models/     # Pydantic schemas
│   │   ├── services/   # Data, model, SHAP, DiCE, explanation_layer
│   │   ├── config.py
│   │   └── main.py
│   ├── artifacts/      # model.joblib, preprocessing.joblib, credit_approval.csv
│   ├── scripts/
│   │   └── train_model.py
│   └── requirements.txt
├── frontend/           # Vite + React
├── README.md
└── ...
```

## Requirements (from Proposal)

| ID  | Requirement |
|-----|-------------|
| FR01 | Upload CSV for analysis |
| FR02 | Decision engine (XGBoost), confidence |
| FR03 | SHAP-based explanations |
| FR04 | Counterfactual what-if exploration |
| FR05 | Clear display of results and explanations |
| FR06 | Export CSV/PDF |
| FR07 | Record user feedback |

## Notes

- **Counterfactuals (DiCE):** The DiCE backend may not initialize with all dataset variants (e.g. some UCI Credit schemas). SHAP and the decision engine still work; counterfactuals are optional.
- **Audit:** Uploads, decisions, and feedback are logged to `backend/logs/audit.jsonl` for reproducibility (NFR07).
- **Schema:** CSV must include columns `A1`–`A15` (UCI Credit schema). Use “Load default” to explore the bundled dataset.
- **Terminology:** **model.predict** → **decision engine**; **accuracy** → **confidence**; **features** → **decision factors**. Everything wraps with **PRISM**.

## License & Ethics

For research use. No personal data collected. See ethical clearance documentation.
