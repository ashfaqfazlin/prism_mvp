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

## Still not working?

1. **Use the frontend URL, not the backend**  
   Open the **prism** (static site) URL in your browser, e.g. `https://prism.onrender.com`. Do **not** open the prism-api URL in the browser; that returns JSON and will show `{"detail": "Not Found"}` at `/`.

2. **URLs don't match the defaults**  
   If your services have different URLs (e.g. `https://prism-abc12.onrender.com`):
   - In **prism-api** → Environment: set **PRISM_CORS_ORIGINS** to your **frontend** URL (exactly as in the address bar).
   - In **prism** → Environment: set **VITE_API_BASE_URL** to your **backend** URL (no trailing slash).
   - **Important:** After changing **VITE_API_BASE_URL**, go to **prism** → **Manual Deploy** and deploy again. The frontend only picks up this value at **build time**.

3. **Check the backend**  
   In a new tab open `https://<your-prism-api-url>/api/health`. You should see `{"status":"ok","app":"PRISM"}`. If you get an error or it takes a long time, the backend may be starting up (free tier cold start ~30–60 s).

4. **Browser console**  
   Open DevTools (F12) → Console. If you see CORS errors, the frontend URL is not in **PRISM_CORS_ORIGINS**. If you see 404 on `/api/...`, the frontend was built without the correct **VITE_API_BASE_URL** — set it and redeploy the frontend (step 2).

## Notes

- **Free tier**: Backend may spin down after ~15 minutes of no traffic; the first request can take 30–60 seconds (cold start).
- **Datasets**: Pre-loaded datasets in `backend/datasets/` are included in the repo and available after deploy. Uploaded files and `study_data/` are **not** persistent on free tier (they’re lost on redeploy or restart).
- **Artifacts**: Trained models in `backend/artifacts/` must be in the repo for decisions and explanations to work on Render. The repo is set up so `backend/artifacts/` is **not** gitignored. If you see "model not trained" for every dataset, add and push artifacts: `git add backend/artifacts/ && git commit -m "Add trained models for Render" && git push`.
