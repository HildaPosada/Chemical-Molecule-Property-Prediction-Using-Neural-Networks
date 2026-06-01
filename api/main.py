"""
FastAPI service for NeuroPass BBB prediction
Wraps the trained PyTorch model for REST API access
"""

import os
import sys
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import MoleculePreprocessor
from src.models import create_model
from src.utils import load_config

# Initialize FastAPI app
app = FastAPI(
    title="NeuroPass API",
    description="Blood-Brain Barrier Penetration Prediction API",
    version="1.0.0"
)

# Configure CORS - allow requests from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None
device = None
config = None


class PredictionRequest(BaseModel):
    """Request body for prediction endpoint"""
    smiles: str

    class Config:
        schema_extra = {
            "example": {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"
            }
        }


class PredictionResponse(BaseModel):
    """Response body for prediction endpoint"""
    smiles: str
    classification: str  # "BBB+" or "BBB-"
    probability: float
    confidence: float

    class Config:
        schema_extra = {
            "example": {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "classification": "BBB+",
                "probability": 0.953,
                "confidence": 0.953
            }
        }


@app.on_event("startup")
async def load_model_on_startup():
    """Load model and preprocessor when API starts"""
    global model, preprocessor, device, config

    try:
        # Load configuration
        config_path = os.getenv("CONFIG_PATH", "config/config_codespaces.yaml")
        config = load_config(config_path)

        # Load preprocessor
        preprocessor = MoleculePreprocessor(config)
        scaler_path = os.path.join(config['data']['processed_dir'], 'scaler.pkl')

        try:
            preprocessor.load_scaler(scaler_path)
        except FileNotFoundError:
            print(f"Warning: Scaler not found at {scaler_path}")

        # Load model
        input_size = preprocessor.get_feature_dim()
        model = create_model(config, input_size)

        device = config['training']['device']
        model_path = os.getenv("MODEL_PATH", "models/checkpoints/best_model.pth")

        checkpoint = torch.load(model_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        print(f"✅ Model loaded successfully from {model_path}")
        print(f"✅ Using device: {device}")

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "NeuroPass API",
        "status": "healthy",
        "version": "1.0.0",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "device": str(device) if device else None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> Dict[str, Any]:
    """
    Predict BBB penetration for a given SMILES string

    Args:
        request: PredictionRequest with SMILES string

    Returns:
        PredictionResponse with classification and probability
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    smiles = request.smiles.strip()

    # Validate SMILES
    if not preprocessor.validate_smiles(smiles):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid SMILES string: {smiles}"
        )

    try:
        # Extract features
        features = preprocessor.extract_features(smiles)
        if features is None:
            raise HTTPException(
                status_code=400,
                detail="Feature extraction failed. Check SMILES string."
            )

        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)

        # Predict
        model.eval()
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)

        # Extract results
        pred_class = int(prediction.item())
        prob_positive = float(probabilities[0, 1].item())
        confidence = float(probabilities[0, pred_class].item())

        # Format response
        classification = "BBB+" if pred_class == 1 else "BBB-"

        return {
            "smiles": smiles,
            "classification": classification,
            "probability": prob_positive,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
