# NeuroPass API

FastAPI service for Blood-Brain Barrier (BBB) penetration prediction.

## API Endpoints

### `GET /`
Health check endpoint

**Response:**
```json
{
  "service": "NeuroPass API",
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true
}
```

### `POST /predict`
Predict BBB penetration for a SMILES string

**Request:**
```json
{
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"
}
```

**Response:**
```json
{
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "classification": "BBB+",
  "probability": 0.953,
  "confidence": 0.953
}
```

**Classification:**
- `BBB+` - Penetrates blood-brain barrier
- `BBB-` - Does not penetrate blood-brain barrier

**Probability:** Probability of BBB+ (0.0 to 1.0)

**Confidence:** Model confidence in the prediction (0.0 to 1.0)

## Local Development

### Install dependencies:
```bash
pip install -r api/requirements.txt
```

### Run the API:
```bash
cd api
python main.py
```

The API will be available at `http://localhost:8000`

### Test the API:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}'
```

## Deployment

This API is configured for deployment to Render using `render.yaml`.

See the main [WEB_APP_GUIDE.md](../WEB_APP_GUIDE.md) for deployment instructions.

## Environment Variables

- `PORT` - Server port (default: 8000)
- `CONFIG_PATH` - Path to model config file (default: config/config_codespaces.yaml)
- `MODEL_PATH` - Path to trained model checkpoint (default: models/checkpoints/best_model.pth)
