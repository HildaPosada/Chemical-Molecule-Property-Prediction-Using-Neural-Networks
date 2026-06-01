"""FastAPI backend for BBB penetration prediction."""

import os
from copy import deepcopy
from functools import lru_cache
from typing import Dict, List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data import MoleculePreprocessor
from src.models import create_model
from src.utils import load_config

APP_BUILD = "2026-06-01"


class PredictRequest(BaseModel):
    """Request payload for single-molecule prediction."""

    smiles: str = Field(..., min_length=1, description="Input SMILES string")


class BatchPredictRequest(BaseModel):
    """Request payload for batch prediction."""

    smiles_list: List[str] = Field(..., min_length=1,
                                   description="List of SMILES strings")


class PredictResponse(BaseModel):
    """Response payload for single-molecule prediction."""

    smiles: str
    classification: str
    probability: float
    confidence_band: str
    prediction: int
    prediction_label: str
    confidence: float
    probability_negative: float
    probability_positive: float
    molecular_properties: Dict[str, float]


class BatchPredictResponse(BaseModel):
    """Response payload for batch prediction."""

    count: int
    results: List[PredictResponse]


class HealthResponse(BaseModel):
    """Health/status response payload."""

    status: str
    model_loaded: bool
    app_build: str


def _parse_allowed_origins() -> List[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _confidence_band(probability: float) -> str:
    """Map model confidence to a human-readable confidence band."""
    if probability >= 0.85:
        return "high"
    if probability >= 0.65:
        return "medium"
    return "low"


def _compute_molecular_properties(smiles: str) -> Dict[str, float]:
    """Compute basic physicochemical properties for explainability."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    return {
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "hbond_donors": float(Descriptors.NumHDonors(mol)),
        "hbond_acceptors": float(Descriptors.NumHAcceptors(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "rotatable_bonds": float(Descriptors.NumRotatableBonds(mol)),
    }


def _get_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
        return checkpoint_obj["model_state_dict"]
    return checkpoint_obj


def _feature_dim(cfg):
    features_cfg = cfg.get("features", {})
    dim = 0
    if features_cfg.get("use_morgan_fingerprints", True):
        dim += int(features_cfg.get("morgan_bits", 1024))
    if features_cfg.get("use_descriptors", True):
        dim += len(features_cfg.get("descriptor_list", []))
    return dim


def _infer_model_overrides(state_dict):
    hidden_sizes = []
    layer_idx = 0
    while f"layers.{layer_idx}.weight" in state_dict:
        hidden_sizes.append(
            int(state_dict[f"layers.{layer_idx}.weight"].shape[0]))
        layer_idx += 1

    input_size = int(state_dict["layers.0.weight"].shape[1])
    num_classes = int(state_dict["output_layer.weight"].shape[0])
    return {
        "input_size": input_size,
        "hidden_sizes": hidden_sizes,
        "num_classes": num_classes,
    }


@lru_cache(maxsize=1)
def get_runtime():
    """Load model and preprocessor once per process."""
    try:
        from rdkit import Chem  # noqa: F401
    except Exception as exc:
        raise RuntimeError("RDKit is required for prediction backend") from exc

    config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
    checkpoint_path = os.getenv(
        "MODEL_CHECKPOINT_PATH", "models/checkpoints/best_model.pth")

    config = load_config(config_path)

    if not os.path.exists(checkpoint_path):
        fallback = "models/saved_models/final_model.pth"
        if os.path.exists(fallback):
            checkpoint_path = fallback
        else:
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path} and fallback {fallback} is missing"
            )

    device = config["training"]["device"]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _get_state_dict(checkpoint)

    if isinstance(checkpoint, dict) and "config" in checkpoint and isinstance(checkpoint["config"], dict):
        config = checkpoint["config"]
        config["training"]["device"] = device
    else:
        checkpoint_input_size = int(state_dict["layers.0.weight"].shape[1])
        if _feature_dim(config) != checkpoint_input_size:
            alt_config_path = "config/config_codespaces.yaml"
            if os.path.exists(alt_config_path):
                alt_config = load_config(alt_config_path)
                if _feature_dim(alt_config) == checkpoint_input_size:
                    config = alt_config

    overrides = _infer_model_overrides(state_dict)
    config = deepcopy(config)
    config["model"]["hidden_sizes"] = overrides["hidden_sizes"]
    config["model"]["num_classes"] = overrides["num_classes"]

    preprocessor = MoleculePreprocessor(config)
    scaler_path = os.path.join(config["data"]["processed_dir"], "scaler.pkl")
    if os.path.exists(scaler_path):
        preprocessor.load_scaler(scaler_path)

    input_size = preprocessor.get_feature_dim()
    if input_size != overrides["input_size"]:
        raise ValueError(
            f"Feature dimension mismatch: preprocessor={input_size}, checkpoint={overrides['input_size']}"
        )

    model = create_model(config, input_size)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, preprocessor, device


def predict_single(smiles: str) -> PredictResponse:
    """Run single-molecule inference."""
    model, preprocessor, device = get_runtime()

    if not preprocessor.validate_smiles(smiles):
        raise HTTPException(status_code=400, detail="Invalid SMILES string")

    features = preprocessor.extract_features(smiles)
    if features is None:
        raise HTTPException(
            status_code=400, detail="Feature extraction failed")

    if preprocessor.scale_features and preprocessor.is_fitted:
        features = preprocessor.scaler.transform(features.reshape(1, -1))[0]

    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)

    pred_idx = int(prediction.item())
    prob_positive = float(probabilities[0, 1].item())
    prob_predicted = float(probabilities[0, pred_idx].item())
    molecular_properties = _compute_molecular_properties(smiles)

    return PredictResponse(
        smiles=smiles,
        classification="BBB+" if pred_idx == 1 else "BBB-",
        probability=prob_positive,
        confidence_band=_confidence_band(prob_predicted),
        prediction=pred_idx,
        prediction_label="Penetrates BBB" if pred_idx == 1 else "Does not penetrate BBB",
        confidence=prob_predicted,
        probability_negative=float(probabilities[0, 0].item()),
        probability_positive=prob_positive,
        molecular_properties=molecular_properties,
    )


app = FastAPI(
    title="NeuroPass Prediction API",
    description="BBB penetration prediction backend for molecular SMILES inputs",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service health check and model readiness."""
    try:
        get_runtime()
        loaded = True
    except Exception:
        loaded = False

    return HealthResponse(status="ok", model_loaded=loaded, app_build=APP_BUILD)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Predict BBB penetration for one SMILES string."""
    try:
        return predict_single(payload.smiles)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/v1/predict", response_model=PredictResponse)
def predict_v1(payload: PredictRequest) -> PredictResponse:
    """Versioned prediction endpoint for stable frontend integration."""
    return predict(payload)


@app.post("/v1/predict-batch", response_model=BatchPredictResponse)
def predict_batch_v1(payload: BatchPredictRequest) -> BatchPredictResponse:
    """Batch prediction endpoint for screening multiple molecules."""
    if len(payload.smiles_list) > 200:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum allowed is 200 molecules per request.",
        )

    results: List[PredictResponse] = []
    for smiles in payload.smiles_list:
        try:
            results.append(predict_single(smiles))
        except HTTPException:
            # Skip invalid entries in batch mode to keep the endpoint useful for
            # large screens; users can inspect successful predictions immediately.
            continue

    return BatchPredictResponse(count=len(results), results=results)
