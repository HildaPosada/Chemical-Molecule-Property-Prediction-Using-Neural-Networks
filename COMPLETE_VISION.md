# NeuroPass Complete Vision

This guide is the source of truth for deploying NeuroPass quickly for demo use and extending to full-stack integration later.

## Current Status

- FastAPI backend exists in [api/main.py](api/main.py)
- Streamlit app exists in [app.py](app.py)
- Render config exists in [render.yaml](render.yaml)
- Frontend (NativelyAI/Vite) is a separate project and must point to the deployed API URL

## Immediate Tasks

1. Merge backend/API changes into main if not already merged.
2. Deploy FastAPI to Render.
3. Make Streamlit app public (if using Streamlit Cloud demo link).
4. Point frontend env var VITE_PREDICT_API_URL to deployed API.

## Deploy FastAPI to Render

1. Go to Render and create a new Blueprint or Web Service from this repo.
2. If using Blueprint, Render will read [render.yaml](render.yaml).
3. Confirm these values:
   - Build command: pip install -r requirements.txt
   - Start command: uvicorn api.main:app --host 0.0.0.0 --port $PORT
4. Environment variables:
   - CONFIG_PATH=config/config.yaml
   - MODEL_CHECKPOINT_PATH=models/checkpoints/best_model.pth
   - ALLOWED_ORIGINS=<your frontend origin or * for demo>
5. Deploy and copy the public API URL.

## API Contract

Endpoint: POST /predict

Request body:
{
  "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
}

Response includes:
- classification (BBB+ or BBB-)
- probability
- smiles

Additional fields are also returned for extended UI use.

## Frontend Integration (Vite)

Set frontend environment variable:

VITE_PREDICT_API_URL=https://your-render-service.onrender.com

Then restart/rebuild frontend.

## Streamlit Cloud (Optional but fast demo)

1. Open share.streamlit.io.
2. Set app to public.
3. Verify public URL loads and can predict.

## Fastest Hackathon Path

1. Make Streamlit public.
2. Demo with Streamlit.
3. Add Render API integration after the hackathon if time is limited.

## Full Platform Path

1. Deploy Render API.
2. Connect Vite frontend to API URL.
3. Keep prediction history in Supabase.
4. Harden CORS and RLS for production.
