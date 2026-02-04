# Deploying PRISM to Render

## Prerequisites

- A [Render](https://render.com) account (free tier is fine)
- This repo pushed to GitHub or GitLab

## 1. Connect and create from Blueprint

1. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**.
2. Connect your Git provider and select this repository.
3. Render will read `render.yaml` and create two services:
   - **prism-api** (backend)
   - **prism** (frontend)
4. When prompted for environment variables, you can leave them blank for now (we’ll set them after the first deploy).

## 2. First deploy

- Let both services deploy. The **backend** will get a URL like `https://prism-api.onrender.com`. The **frontend** will get a URL like `https://prism.onrender.com` (or similar).

## 3. Set environment variables

### Backend (prism-api)

1. Open the **prism-api** service → **Environment**.
2. Add or edit:
   - **PRISM_CORS_ORIGINS** = your frontend URL (no trailing slash), e.g. `https://prism.onrender.com`
3. Save. Render will redeploy the backend.

### Frontend (prism)

1. Open the **prism** service → **Environment**.
2. Add or edit:
   - **VITE_API_BASE_URL** = your backend URL (no trailing slash), e.g. `https://prism-api.onrender.com`
3. Save. Render will redeploy the frontend (the new build will use this URL for API calls).

## 4. Verify

1. Open the frontend URL in a browser.
2. Select a dataset and run a decision. If the app loads and decisions work, deployment is correct.

## Notes

- **Free tier**: Backend may spin down after ~15 minutes of no traffic; the first request can take 30–60 seconds (cold start).
- **Datasets**: Pre-loaded datasets in `backend/datasets/` are included in the repo and available after deploy. Uploaded files and `study_data/` are **not** persistent on free tier (they’re lost on redeploy or restart).
- **Artifacts**: Trained models in `backend/artifacts/` must be in the repo (or built at deploy time) for domain-specific models to work. Ensure `backend/artifacts/` is committed or add a build step to train/copy models.
