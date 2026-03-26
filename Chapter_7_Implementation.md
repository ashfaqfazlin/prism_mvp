# Chapter 7: Implementation

This chapter describes how the PRISM system was implemented from the design presented in Chapter 6. It covers technology selection (with justification for each choice), core functionalities implementation (including dataset usage and modular structure), user interface implementation, challenges encountered and solutions applied, and a chapter summary. The content aligns with the Software Implementation lectures (Week 10 and Week 11) and ensures that every software, framework, and tool selection is justified. **All datasets and any third-party or referenced code used in the implementation must be cited; uncited code or data is treated as plagiarism and can result in zero marks for the FYP module.**

---

## 7.1 Chapter Overview

The implementation phase transforms the design into a working prototype. The objectives of this phase were to: (1) implement the decision engine (black-box classifier) with preprocessing and inference; (2) implement SHAP-based and counterfactual explainability and the plain-language explanation layer; (3) expose a REST API for the frontend; (4) build a React-based user interface for dataset selection, decision viewing, explanation modes, and export; and (5) support multiple domains (catalog and user uploads) with consistent behaviour. This chapter provides a roadmap that links the design (Chapter 6) to the implemented system: Section 7.2 justifies the technology stack; Section 7.3 details core module implementation and how it relates to the SRS; Section 7.4 describes the UI and backend integration; Section 7.5 documents challenges and solutions to demonstrate the effort invested in workarounds and parameter discovery; Section 7.6 summarises the implementation and readiness for testing.

---

## 7.2 Technology Selection

The tools, frameworks, and programming languages were chosen to meet the software requirements specification (SRS), support the AI/ML pipeline and explainability, and enable maintainable development. Each selection is justified below.

### 7.2.1 Technology Stack

The technology stack is summarised in the following table. The rest of this section elaborates on each category.

| Layer | Technology | Version (example) | Purpose |
|-------|------------|-------------------|---------|
| Backend language | Python | 3.10+ | ML, data processing, API |
| Web framework | FastAPI | 0.110+ | REST API, validation, async |
| ML framework | scikit-learn, XGBoost | 1.5+, 2.0+ | Preprocessing, classification |
| Explainability | SHAP, DiCE | 0.44+, 0.11+ | Feature attribution, counterfactuals |
| Frontend | React | 19.x | UI components, state |
| Build / dev (frontend) | Vite | 7.x | Dev server, proxy, build |
| Data / export | Pandas, ReportLab | 2.2+, 4.0+ | Data handling, PDF export |
| Serialisation | joblib | 1.3+ | Model and pipeline persistence |

*A full visualisation of the technology stack (e.g. a diagram with tiers) can be included here or in the appendix.*

### 7.2.2 Programming Languages

- **Python (3.10+)**  
  Python was selected for the backend and ML pipeline because it is the de facto language for data science and machine learning (Pedregosa et al., 2011; Chen & Guestrin, 2016). It provides mature libraries for preprocessing (scikit-learn), gradient boosting (XGBoost), explainability (SHAP, DiCE), and data manipulation (Pandas, NumPy), all of which are required by the SRS. Python’s readability and the use of type hints and Pydantic support maintainability and alignment with the design’s service-oriented structure.

- **JavaScript (ES6+) / JSX**  
  The frontend is implemented in JavaScript (React with JSX) to support a rich, single-page interface with components for the data table, decision panel, explanation modes, and export. React’s component model and hooks (e.g. `useState`, `useCallback`) match the need for local state, API calls, and responsive updates without full page reloads, supporting the usability and responsiveness NFRs.

### 7.2.3 Development Framework

- **FastAPI**  
  FastAPI was chosen as the backend framework (FastAPI, n.d.) because it offers automatic OpenAPI documentation, request/response validation via Pydantic, and asynchronous support, which helps meet the performance and maintainability requirements. It integrates cleanly with the existing Python service layer and allows clear separation between API routes and business logic (as in Chapter 6). CORS is configurable for the frontend origin (e.g. `http://localhost:5173` during development).

- **React**  
  React was selected for the frontend (React, n.d.) because it allows a modular, component-based UI that maps to the design’s presentation tier. The single-page application (SPA) structure keeps the user in one context (dataset, row, decision, explanations) and supports the three explanation modes (plain language, technical SHAP, what-if) and export without navigating away. This aligns with the usability and human-centred design goals.

- **Vite**  
  Vite is used as the frontend build tool and dev server (Vite, n.d.). It provides fast hot module replacement (HMR) and a simple proxy configuration so that all `/api` requests from the frontend are forwarded to the backend (e.g. `http://127.0.0.1:8000`), avoiding cross-origin issues during development and keeping the frontend code independent of the backend base URL (via `import.meta.env.VITE_API_BASE_URL` when needed for production).

### 7.2.4 Libraries and Toolkits

- **scikit-learn**  
  Used for data preprocessing (e.g. `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`) and train/test splitting (Pedregosa et al., 2011). It ensures that the same preprocessing applied at training time is applied at inference time, supporting the accuracy and consistency requirements.

- **XGBoost**  
  Used as the black-box classifier (decision engine) (Chen & Guestrin, 2016). XGBoost was chosen for its strong performance on tabular data and compatibility with SHAP’s TreeExplainer, which provides exact SHAP values for tree-based models and supports the explainability NFR.

- **SHAP**  
  Used to compute feature attributions (SHAP values) for each prediction (Lundberg & Lee, 2017; SHAP, n.d.). TreeExplainer is used with a small background sample (e.g. 100 rows) to balance explanation fidelity and response time, supporting both accuracy and performance.

- **DiCE (dice-ml)**  
  Used for counterfactual (“what-if”) generation where supported (Mothilal et al., 2020; DiCE, n.d.). It was selected because it is model-agnostic and can wrap the same sklearn pipeline used for prediction, keeping what-if outcomes consistent with the model. Not all dataset schemas are supported by DiCE; where it fails, SHAP and the decision engine remain functional.

- **Pandas / NumPy**  
  Used for loading CSV datasets, validating uploads, and preparing feature matrices (McKinney, 2010; Harris et al., 2020). Pandas supports the data handling required by the SRS (catalog datasets, uploads, and export).

- **Pydantic**  
  Used for API request and response schemas (e.g. `DecisionResponse`, `ShapValuesResponse`, `ExportRequest`) (Pydantic, n.d.). This ensures validation at the API boundary and consistent contracts between frontend and backend, supporting reliability and maintainability.

- **ReportLab**  
  Used for generating PDF exports of decisions and explanations, supporting the auditability and export requirements of the SRS (ReportLab, n.d.).

- **joblib**  
  Used to persist fitted preprocessing pipelines and trained models per domain, so that the same artifacts can be loaded at runtime without retraining (joblib, n.d.; scikit-learn uses joblib for model persistence).

- **Recharts**  
  Used in the frontend to render the SHAP bar chart (decision factors vs. impact) (Recharts, n.d.). It was chosen for its React integration and ability to produce accessible, responsive charts that support the technical explanation mode.

*All library versions and usage in the codebase should be cited appropriately in the references section of the thesis.*

### 7.2.5 Integrated Development Environment (IDEs)

- **Visual Studio Code / PyCharm**  
  Development was carried out using IDEs such as Visual Studio Code or PyCharm. These support Python debugging, React/JSX editing, terminal integration for running the backend (e.g. `uvicorn app.main:app --reload`) and frontend (`npm run dev`), and version control (Git), which helps meet the maintainability and reproducibility goals. The choice of IDE does not affect the runtime behaviour of the system.

### 7.2.6 Summary of the Technology Selection

The technology stack was chosen to satisfy the SRS and design: Python and scikit-learn/XGBoost for the ML pipeline; SHAP and DiCE for explainability; FastAPI for a validated, documented API; React and Vite for a responsive, maintainable frontend; and joblib/ReportLab for persistence and export. Each selection was justified in terms of relevance to the software requirements, implementation support, and alignment with Chapter 6. The same stack is used consistently across the implementation described in Sections 7.3 and 7.4.

---

## 7.3 Core Functionalities Implementation

This section explains how the core features of PRISM were developed and integrated. It includes dataset declaration and train/test split statistics, the implementation of each major module (related to the functional requirements in the SRS), the logical organisation of the code, and how modules interact. **Any code or dataset taken from external sources must be cited; failure to cite can result in zero for the FYP module.**

### 7.3.1 Dataset Declaration and Train/Test Split

PRISM uses **publicly available datasets** for catalog domains. The primary sources are the **UCI Machine Learning Repository** (Dua & Graff, 2019). Datasets are declared in the project as follows:

- **Catalog datasets** are listed in `backend/datasets/catalog.json` with attributes such as `id`, `name`, `source`, `filename`, `rows`, `features`, `target`, and (where applicable) reported `accuracy`. Examples include:
  - UCI Credit Approval (`uci_credit_approval.csv`), German Credit Risk (`german_credit.csv`), Taiwan Credit Card Default, Heart Disease, Diabetes, Breast Cancer, Bank Marketing, Student Performance, HR Attrition, Insurance (COIL), and Recidivism (COMPAS).
- **Training and evaluation** use an 80–20 stratified train/test split with a fixed random seed (e.g. `random_state=42`) for reproducibility. For example, in `train_all_models.py`:
  - `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`.
- **Statistics** (e.g. number of features, samples, target distribution, encoded feature count, test accuracy) are logged during training and stored in `artifacts/<domain_id>/meta.json` (e.g. `test_accuracy`, `feature_cols`, `encoded_feature_names`). Table 1 below reports, for each catalog domain: dataset name, source, total rows, training size, test size (80–20 stratified split), number of input features, and test accuracy from the saved artifacts. *All dataset sources must be cited; see the References section and dataset-specific citations below.*

**Table 1: Dataset declaration and train/test split statistics (catalog domains)**

| Dataset name | Source | Total rows | Train size | Test size | No. features | Test accuracy |
|--------------|--------|------------|------------|-----------|--------------|---------------|
| UCI Credit Approval | UCI ML Repository (Dua & Graff, 2019) | 690 | 552 | 138 | 15 | 0.891 |
| German Credit Risk | UCI ML Repository – Statlog (Dua & Graff, 2019) | 1,000 | 800 | 200 | 20 | 0.775 |
| Taiwan Credit Card Default | UCI ML Repository (Yeh & Lien, 2009) | 30,000 | 24,000 | 6,000 | 23 | 0.818 |
| Bank Marketing | UCI ML Repository (Moro et al., 2014) | 41,188 | 32,950 | 8,238 | 20 | 0.923 |
| Heart Disease Prediction | UCI ML Repository – Cleveland (Janosi et al., 1988) | 303 | 242 | 61 | 13 | 0.852 |
| Diabetes Prediction | NIDDK Pima Indians (Smith et al., 1988) | 768 | 614 | 154 | 8 | 0.747 |
| Breast Cancer Diagnosis | UCI ML Repository – Wisconsin (Wolberg & Mangasarian, 1990) | 569 | 455 | 114 | 30 | 0.974 |
| Student Performance | UCI ML Repository (Cortez & Silva, 2008) | 395 | 316 | 79 | 30 | 0.633 |
| Employee Attrition | IBM HR Analytics / Kaggle (IBM, 2018) | 1,470 | 1,176 | 294 | 30 | 0.864 |
| Caravan Insurance | UCI ML Repository – COIL 2000 (van der Putten & van Someren, 2004) | 5,822 | 4,658 | 1,164 | 85 | 0.940 |
| Recidivism Prediction | ProPublica COMPAS (Angwin et al., 2016) | 7,214 | 5,771 | 1,443 | 8 | 0.685 |

*Note:* Train and test sizes are based on an 80–20 stratified split with `random_state=42`. Test accuracy is read from `artifacts/<domain_id>/meta.json` after training. All datasets are used in accordance with their respective licences and cited in the References.

Optional **exploratory data analysis (EDA)** (e.g. missing values, class balance, feature distributions) can be conducted before training; if performed, it should be briefly described and any figures or statistics included with citations for the dataset.

### 7.3.2 Implementation of Modules and Relation to Functional Requirements

The implementation is organised into modules that map to the functional requirements (FRs) defined in the SRS (Chapter 4). Below, each major module is summarised and linked to the design and SRS.

| Module | Location | Functionality | Relation to SRS / Design |
|--------|----------|---------------|---------------------------|
| **Domain configuration** | `app/domain_config.py` | Defines per-domain schema (features, target, labels), artifact paths, and a registry of domains. | Supports FRs for multi-dataset support and explainability labels. |
| **Data service** | `app/services/data_service.py`, `dataset_service.py` | Loads and validates CSV data; provides preprocessing and feature names; manages catalog and recent uploads metadata. | Supports data ingestion, validation, and upload handling (FRs). |
| **Model service** | `app/services/model_service.py`, `domain_model_service.py` | Loads trained pipeline (preprocessing + classifier) from disk; runs prediction and returns decision, confidence, probabilities. | Implements the decision engine (FR: prediction for a row). |
| **Explainability service** | `app/services/explainability_service.py`, `domain_explainability_service.py` | Builds SHAP TreeExplainer (with background sample); computes SHAP values; optionally integrates DiCE for counterfactuals. | Implements SHAP and counterfactual explanations (FRs). |
| **Explanation layer** | `app/services/explanation_layer.py` | Converts SHAP output and row data into plain-language bullets, directional reasoning, and decision-factor summary using feature labels and units. | Implements human-readable explanations (FR: plain-language mode). |
| **API routes** | `app/api/routes.py` | Exposes REST endpoints for catalog, upload, prediction, explain, export; validates input/output with Pydantic. | Implements the API contract linking frontend to services (design Chapter 6). |
| **Training scripts** | `scripts/train_all_models.py`, `train_model.py` | Load dataset, preprocess, train XGBoost, evaluate, persist pipeline and metadata. | Implements the training pipeline (design Section 6.5). |
| **Dynamic domain / auto-trainer** | `app/services/dynamic_domain_service.py`, `auto_trainer_service.py`, `auto_schema_service.py` | Create domains from user uploads; infer schema; train and persist models for custom datasets. | Supports upload-based domains and custom training (FRs). |

### 7.3.3 Code Structure and Design Patterns

The code is organised using **classes and singleton instances** for services, **Pydantic models** for API contracts, and **configuration** (e.g. `app/config.py`) for paths and limits. Key patterns:

- **Singleton services:** e.g. `model_service`, `data_service`, `dataset_service` are instantiated once and reused so that models and explainers can be loaded and cached per domain.
- **Lazy loading:** Models and SHAP explainers are loaded on first use (`_ensure_loaded()`) to avoid startup delay and to support domains that may not all be used in a single session.
- **Separation of concerns:** API layer only validates and delegates; business logic lives in services; configuration and domain metadata are separate from execution.

**Code snippet (example) — Decision engine prediction**

The following illustrates how the model service exposes a single-instance prediction. The logic uses the loaded pipeline (preprocessing + classifier) and returns a decision string, confidence, and class probabilities. *This code is part of the PRISM implementation; any adapted or referenced code from documentation or tutorials must be cited.*

```python
# app/services/model_service.py (simplified representation)
def predict_single(self, x: np.ndarray) -> tuple[str, float, dict[str, float]]:
    """Single instance -> (decision, confidence, {class: prob})."""
    x = x.reshape(1, -1)
    labels, probs = self.predict(x)
    lab = int(labels[0])
    pred = "+" if lab == 1 else "-"
    conf = float(probs[0, lab])
    probs_d = {"+": float(probs[0, 1]), "-": float(probs[0, 0])}
    return pred, conf, probs_d
```

The API layer receives a row (and domain_id for multi-domain), calls the appropriate service to preprocess and predict, and returns a `DecisionResponse` (Pydantic) to the frontend.

**Code snippet (example) — Plain-language explanation**

The explanation layer takes SHAP results and the current row and produces bullets and directional reasoning. It uses feature labels and value formatting so that explanations align with the human-centred design (Chapter 6). *Feature labels and formatting logic are project-specific; any reuse of external explanation templates or text should be cited.*

```python
# app/services/explanation_layer.py (conceptual)
def plain_language_explanations(shap_result, decision, row, top_k=5):
    # Rank factors by |SHAP|; separate positive vs negative contribution
    # Build bullets: e.g. "Your {label} ({value}) positively influenced the approval."
    # Build directional sentence: e.g. "Key factors like {top_positive} contributed positively."
    return {"bullets": bullets, "summary": summary, "directional_reasoning": directional}
```

Screenshots of further code snippets (e.g. training loop, SHAP call, API route) can be included in the thesis with accompanying explanation and citations where applicable.

### 7.3.4 Integration of Modules

- **Request flow:** Frontend calls e.g. `POST /api/predict` and `GET /api/explain` (or domain-specific equivalents). The API loads the correct domain via `DomainModelService` / `DomainExplainabilityService`, runs preprocessing and prediction, then SHAP and (optionally) the explanation layer and counterfactuals. Responses are JSON (or file stream for export).
- **Data flow:** Catalog datasets are read from `datasets/`; trained artifacts from `artifacts/<domain_id>/`; recent uploads metadata from `app_data/recent_uploads.json`. No full upload content is persisted beyond the recent-uploads list (design and SLEP Chapter 5).
- **Consistency:** The same preprocessing pipeline and model used at training (in `train_all_models.py`) are loaded at inference, so that feature encoding and scaling are identical. SHAP uses the same pipeline and feature names as the model service, ensuring explanation fidelity.

---

## 7.4 User Interface Implementation

This section highlights the development of the user interface and its integration with the backend, including accessibility considerations.

### 7.4.1 Frontend Development

The UI is implemented as a **single-page React application** (e.g. `frontend/src/App.jsx`). Main elements:

- **Input forms and controls:** Dataset picker (catalog dropdown and recent uploads), CSV upload button, row limit and search/filter for the data table. For custom uploads, a training wizard (analyze → configure target → train) is provided.
- **Dashboards and result visualisations:**  
  - **Data table:** Displays rows for the selected dataset; single-row selection triggers prediction and explanation.  
  - **Decision panel:** Shows decision (e.g. Approved/Rejected), confidence (e.g. bar or percentage), and class probabilities.  
  - **Explanation area:** Three modes—plain language (bullets and directional text), technical (SHAP bar chart via Recharts), and what-if (sliders for a subset of features, updated prediction, optional counterfactual suggestions).  
  - **Saved cases:** List of bookmarked (dataset, row) pairs; clicking a case loads it and scrolls to the explanation.  
  - **Export:** Buttons or menu to download CSV or PDF reports for the current case or bulk export.
- **Theme toggle:** Light/dark theme stored in `localStorage` and applied globally to support usability and accessibility (contrast and preference).

The structure uses React state (e.g. `useState` for rows, selected index, result, explanation mode, bookmarks, theme) and effects (e.g. `useEffect` or `useCallback` for loading catalog, dataset, and explanation when the selection changes). Components are organised within a single `App` component; for a larger codebase they could be split into presentational components (e.g. `DecisionPanel`, `ExplanationView`, `DataTable`).

### 7.4.2 Backend Integration

- **API client:** The frontend uses a central `api.js` module that defines a base URL (e.g. `/api` when using the Vite proxy, or `VITE_API_BASE_URL` for production) and helper functions for each endpoint (e.g. `getDatasetCatalog`, `loadDataset`, `requestDomainDecision`, `requestBatchPredictions`, `exportReport`, `exportBulk`). All requests use `fetch` with JSON headers and error handling that surfaces backend messages.
- **Proxy:** During development, Vite is configured to proxy `/api` to `http://127.0.0.1:8000`, so the frontend runs on e.g. `http://localhost:5173` and API calls are sent to the same origin, avoiding CORS issues. Production deployment can set `VITE_API_BASE_URL` to the deployed backend URL.
- **Data flow:** On dataset load, the frontend fetches rows and (where applicable) feature ranges; on row selection it requests prediction and then explanation; on what-if slider change it may request a new prediction or counterfactuals; on export it sends the current (or bulk) data and receives a file stream. Loading and error states are shown so that the user receives clear feedback.

### 7.4.3 Accessibility Considerations

- **Themes:** Light and dark themes improve readability and meet user preference; sufficient contrast should be ensured (e.g. WCAG AA) in both themes.
- **Structure:** Semantic structure (e.g. headings, regions) and clear labels for controls (dataset picker, sliders, export buttons) support screen readers and keyboard navigation.
- **Charts:** The SHAP bar chart is complemented by structured data (e.g. decision-factor names and values) so that key information is not conveyed by colour or shape alone; this supports accessibility and aligns with the design (Chapter 6).

---

## 7.5 Challenges and Solutions

This section documents obstacles encountered during implementation and how they were addressed. It is important to show the effort invested in discovering workarounds, obtaining parameter values, and completing the implementation (as emphasised in the module guidance).

### 7.5.1 DiCE and Schema Compatibility

- **Challenge:** The DiCE library (dice-ml) expects specific data formats and model interfaces. For some dataset schemas (e.g. mixed types, many categories, or certain encodings), DiCE failed to generate counterfactuals or raised errors, which would have broken the what-if experience if counterfactuals were mandatory.
- **Solution:** DiCE integration was made optional. The explainability service tries to build the DiCE explainer; on failure it sets the explainer to `None` and continues. The API and frontend handle missing counterfactuals gracefully: SHAP and the decision engine still work; the what-if view can rely on sliders to request new predictions for modified rows instead of DiCE-generated suggestions. Parameter choices (e.g. DiCE backend, number of counterfactuals) were tuned by consulting the library documentation (Mothilal et al., 2020; DiCE, n.d.) and testing on the UCI Credit and German Credit datasets (Dua & Graff, 2019).

### 7.5.2 SHAP Performance and Background Sample Size

- **Challenge:** SHAP values can be slow for large background samples or complex models. A large background set increased response time and conflicted with the performance NFR.
- **Solution:** The background sample size for TreeExplainer was limited (e.g. to 100 rows from the domain dataset). This was chosen after testing: it kept explanations stable enough for the interface while reducing latency. The value is documented in the code (e.g. in `domain_explainability_service.py`) so that it can be adjusted if needed (Lundberg & Lee, 2017; SHAP, n.d.).

### 7.5.3 Multi-Domain Model and Explainer Caching

- **Challenge:** Loading a new model and building a new SHAP explainer for every request would have been too slow when switching between domains or when many users (or many rows) were served.
- **Solution:** Domain-specific models and explainers are cached in memory. `DomainModelService` and `DomainExplainabilityService` maintain a current domain and (for explainers) a dictionary keyed by `domain_id`. On domain switch, the correct model and explainer are loaded once and reused. Cache invalidation is not required for the current scope because catalog models and upload-trained models are static until retrained.

### 7.5.4 Reproducibility and Hyperparameters

- **Challenge:** Results needed to be reproducible for evaluation and documentation. Hyperparameter choices (e.g. XGBoost `n_estimators`, `max_depth`, `learning_rate`) and train/test split had to be fixed and documented.
- **Solution:** A fixed `random_state=42` was used for `train_test_split` and for the XGBoost classifier. XGBoost parameters (e.g. `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`) were set in `train_all_models.py` after comparing a small set of options; the chosen values are documented in code and in the thesis. This supports reproducibility and the ability to report consistent accuracy and explanation behaviour.

### 7.5.5 CORS and Frontend–Backend Connection

- **Challenge:** During development, the frontend (e.g. `localhost:5173`) and backend (e.g. `localhost:8000`) run on different origins. Browsers block cross-origin requests unless CORS headers allow it, which could prevent the UI from receiving API responses.
- **Solution:** The backend was configured with CORS middleware (FastAPI `CORSMiddleware`) allowing the frontend origins (e.g. `http://localhost:5173`, `http://127.0.0.1:5173`). In development, the Vite proxy was also used so that the frontend can call `/api` on the same origin and Vite forwards to the backend. For production, the deployed backend must list the deployed frontend origin in `cors_origins` (or equivalent config).

### 7.5.6 Upload Validation and Row Limits

- **Challenge:** Large or malformed CSV uploads could cause timeouts, memory issues, or validation errors that were unclear to the user.
- **Solution:** Configurable limits were introduced: maximum file size (e.g. 50 MB) and maximum number of rows (e.g. 50,000). The API returns clear error messages (e.g. "File too large", "Missing columns: ..."). Validation uses the same preprocessing schema where applicable so that only valid feature columns are accepted. Recent uploads store only metadata and a bounded sample of rows (e.g. 50) to limit storage and protect against accidental retention of large datasets.

---

## 7.6 Chapter Summary

This chapter described the implementation of the PRISM system. **Section 7.1** outlined the objectives of the implementation phase and linked them to the design (Chapter 6). **Section 7.2** justified the technology selection: programming languages (Python, JavaScript/JSX), development frameworks (FastAPI, React, Vite), and libraries (scikit-learn, XGBoost, SHAP, DiCE, Pandas, Pydantic, ReportLab, Recharts, joblib), with reasons tied to the SRS and design. **Section 7.3** covered core functionalities: dataset declaration and train/test split statistics, implementation of each major module and its relation to the SRS, code structure (classes, singletons, lazy loading), and integration between API, services, and data. **Section 7.4** described the user interface implementation (frontend components, backend integration via API client and proxy, and accessibility considerations). **Section 7.5** documented challenges and solutions—DiCE schema compatibility, SHAP performance and background sample size, multi-domain caching, reproducibility and hyperparameters, CORS and proxy setup, and upload validation—to demonstrate the effort put into workarounds and parameter discovery. The implementation is complete and ready for testing (Chapter 8). **All datasets and any code or libraries taken from external sources must be properly cited in the thesis references to avoid plagiarism.**

---

## References

**Software and libraries**

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785
- Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, *585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)* (pp. 4765–4774). https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
- McKinney, W. (2010). Data structures for statistical computing in Python. In S. van der Walt & J. Millman (Eds.), *Proceedings of the 9th Python in Science Conference (SciPy 2010)* (pp. 56–61). https://doi.org/10.25080/Majora-92bf1922-00a
- Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations. In *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency (FAT* ’20)* (pp. 607–617). ACM. https://doi.org/10.1145/3351095.3372850
- Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
- Pydantic. (n.d.). *Pydantic: Data validation using Python type annotations*. https://docs.pydantic.dev/
- ReportLab. (n.d.). *ReportLab Open Source*. https://www.reportlab.com/opensource/
- Recharts. (n.d.). *Recharts – Composable charting library built on React*. https://recharts.org/
- FastAPI. (n.d.). *FastAPI framework*. https://fastapi.tiangolo.com/
- React. (n.d.). *React – A JavaScript library for building user interfaces*. https://react.dev/
- Vite. (n.d.). *Vite – Next generation frontend tooling*. https://vitejs.dev/
- SHAP. (n.d.). *SHAP (SHapley Additive exPlanations)*. https://github.com/slundberg/shap
- DiCE. (n.d.). *dice-ml: Diverse Counterfactual Explanations*. https://github.com/interpretml/DiCE
- joblib. (n.d.). *joblib: Running Python functions as pipeline jobs*. https://joblib.readthedocs.io/

**Datasets**

- Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016). Machine bias: There’s software used across the country to predict future criminals. And it’s biased against blacks. *ProPublica*. https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing (COMPAS recidivism data).
- Cortez, P., & Silva, A. M. G. (2008). Using data mining to predict secondary school student performance. In A. Brito & J. Teixeira (Eds.), *Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008)* (pp. 5–12). EUROSIS. (Student Performance; UCI ML Repository.)
- Dua, D., & Graff, C. (2019). *UCI Machine Learning Repository*. University of California, Irvine, School of Information and Computer Sciences. https://archive.ics.uci.edu/ml
- IBM. (2018). *IBM HR Analytics Employee Attrition & Performance*. Kaggle. https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset (or current URL).
- Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). *Heart Disease*. UCI ML Repository. https://archive.ics.uci.edu/ml/datasets/Heart+Disease (Cleveland dataset).
- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, *62*, 22–31. https://doi.org/10.1016/j.dss.2014.03.001 (Bank Marketing; UCI ML Repository).
- Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In *Proceedings of the Symposium on Computer Applications and Medical Care* (pp. 261–265). IEEE. (Pima Indians Diabetes; NIDDK / UCI.)
- van der Putten, P., & van Someren, M. (2004). A bias-variance analysis of a real-world learning problem: The COIL challenge 2000. *Machine Learning*, *57*(1–2), 177–195. https://doi.org/10.1023/B:MACH.0000035477.14468.42 (COIL 2000; UCI ML Repository).
- Wolberg, W. H., & Mangasarian, O. L. (1990). Multisurface method of pattern separation for medical diagnosis applied to breast cytology. *Proceedings of the National Academy of Sciences*, *87*(23), 9193–9196. https://doi.org/10.1073/pnas.87.23.9193 (Breast Cancer Wisconsin; UCI ML Repository).
- Yeh, I.-C., & Lien, C.-h. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, *36*(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020 (Taiwan Credit Card Default; UCI ML Repository.)

*Formatting (e.g. Harvard, APA) can be adjusted to match your institution’s thesis guidelines. URLs and access dates may be added where required.*
