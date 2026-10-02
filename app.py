"""
NeuroPass - AI-Powered Blood-Brain Barrier Penetration Prediction
Streamlit Web Interface
"""

from src.utils import load_config
from src.models import create_model
from src.data import MoleculePreprocessor
import os
import json
from pathlib import Path
import sys
import streamlit as st
import torch
import pandas as pd
from PIL import Image
from io import BytesIO
import base64
from copy import deepcopy

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

APP_BUILD = "2026-10-01-comparison"


# Try to import RDKit chemistry modules.
# Drawing modules are imported lazily because they can be unavailable in some
# deployments even when core RDKit chemistry support is installed.
RDKIT_CHEM_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_CHEM_AVAILABLE = True
except ImportError:
    RDKIT_CHEM_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="NeuroPass - BBB Penetration Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .prediction-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .prediction-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model_and_preprocessor():
    """Load the trained model and preprocessor."""
    try:
        if not RDKIT_CHEM_AVAILABLE:
            raise ImportError(
                "RDKit chemistry modules are not available. Install RDKit to enable prediction."
            )

        def _get_state_dict(checkpoint_obj):
            if isinstance(checkpoint_obj, dict) and 'model_state_dict' in checkpoint_obj:
                return checkpoint_obj['model_state_dict']
            return checkpoint_obj

        def _feature_dim(cfg):
            features_cfg = cfg.get('features', {})
            dim = 0
            if features_cfg.get('use_morgan_fingerprints', True):
                dim += int(features_cfg.get('morgan_bits', 1024))
            if features_cfg.get('use_descriptors', True):
                dim += len(features_cfg.get('descriptor_list', []))
            return dim

        def _infer_model_overrides(state_dict):
            hidden_sizes = []
            layer_idx = 0
            while f'layers.{layer_idx}.weight' in state_dict:
                hidden_sizes.append(
                    int(state_dict[f'layers.{layer_idx}.weight'].shape[0]))
                layer_idx += 1

            input_size = int(state_dict['layers.0.weight'].shape[1])
            num_classes = int(state_dict['output_layer.weight'].shape[0])
            return {
                'input_size': input_size,
                'hidden_sizes': hidden_sizes,
                'num_classes': num_classes,
            }

        # Load base configuration and checkpoint
        config = load_config('config/config.yaml')
        checkpoint_path = 'models/checkpoints/best_model.pth'
        if not os.path.exists(checkpoint_path):
            checkpoint_path = 'models/saved_models/final_model.pth'

        device = config['training']['device']
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = _get_state_dict(checkpoint)

        # Prefer checkpoint-embedded config when available.
        # Otherwise, fall back to a known codespaces config if dimensions don't match.
        if isinstance(checkpoint, dict) and 'config' in checkpoint and isinstance(checkpoint['config'], dict):
            config = checkpoint['config']
            config['training']['device'] = device
        else:
            checkpoint_input_size = int(state_dict['layers.0.weight'].shape[1])
            if _feature_dim(config) != checkpoint_input_size:
                alt_config_path = 'config/config_codespaces.yaml'
                if os.path.exists(alt_config_path):
                    alt_config = load_config(alt_config_path)
                    if _feature_dim(alt_config) == checkpoint_input_size:
                        config = alt_config

        # Force model section to match checkpoint architecture
        overrides = _infer_model_overrides(state_dict)
        config = deepcopy(config)
        config['model']['hidden_sizes'] = overrides['hidden_sizes']
        config['model']['num_classes'] = overrides['num_classes']

        # Load preprocessor
        preprocessor = MoleculePreprocessor(config)
        scaler_path = os.path.join(
            config['data']['processed_dir'], 'scaler.pkl')

        try:
            preprocessor.load_scaler(scaler_path)
        except FileNotFoundError:
            st.warning(
                "Scaler not found. Predictions will use non-scaled features.")

        # Load model
        input_size = preprocessor.get_feature_dim()
        if input_size != overrides['input_size']:
            raise ValueError(
                f"Feature dimension mismatch: preprocessor={input_size}, checkpoint={overrides['input_size']}. "
                "Please use the training-time config/checkpoint pair."
            )

        model = create_model(config, input_size)
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        return model, preprocessor, device, config

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None


def predict_molecule(smiles: str, model, preprocessor, device):
    """Make prediction for a single SMILES string."""
    # Validate SMILES
    if not preprocessor.validate_smiles(smiles):
        return {
            'valid': False,
            'error': 'Invalid SMILES string'
        }

    # Extract features
    features = preprocessor.extract_features(smiles)
    if features is None:
        return {
            'valid': False,
            'error': 'Feature extraction failed'
        }

    # Apply training-time feature scaling if scaler is available
    if preprocessor.scale_features and preprocessor.is_fitted:
        features = preprocessor.scaler.transform(features.reshape(1, -1))[0]

    # Convert to tensor
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)

    predicted_idx = int(prediction.item())
    predicted_confidence = float(probabilities[0, predicted_idx].item())
    if predicted_confidence >= 0.85:
        confidence_band = 'High confidence'
    elif predicted_confidence >= 0.65:
        confidence_band = 'Medium confidence'
    else:
        confidence_band = 'Low confidence'

    return {
        'valid': True,
        'prediction': predicted_idx,
        'prediction_label': 'Penetrates BBB' if predicted_idx == 1 else 'Does not penetrate BBB',
        'confidence': predicted_confidence,
        'confidence_band': confidence_band,
        'probability_negative': float(probabilities[0, 0].item()),
        'probability_positive': float(probabilities[0, 1].item())
    }


def draw_molecule(smiles: str):
    """Render molecular structure as SVG without requiring the Cairo PNG backend."""
    if not RDKIT_CHEM_AVAILABLE:
        return None

    try:
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Molecular SVG rendering failed")
        return None


def get_molecular_properties(smiles: str):
    """Calculate basic molecular properties."""
    if not RDKIT_CHEM_AVAILABLE:
        return {}

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        properties = {
            'Molecular Weight': f"{Descriptors.MolWt(mol):.2f} g/mol",
            'LogP': f"{Descriptors.MolLogP(mol):.2f}",
            'H-Bond Donors': str(Descriptors.NumHDonors(mol)),
            'H-Bond Acceptors': str(Descriptors.NumHAcceptors(mol)),
            'Rotatable Bonds': str(Descriptors.NumRotatableBonds(mol)),
            'Aromatic Rings': str(Descriptors.NumAromaticRings(mol)),
            'TPSA': f"{Descriptors.TPSA(mol):.2f} A^2"
        }
        return properties
    except Exception as e:
        return {}


def main():
    """Main Streamlit application."""

    if not RDKIT_CHEM_AVAILABLE:
        st.error(
            "RDKit is not available in this environment. Prediction and molecular analysis require RDKit."
        )
        st.stop()

    # Header
    st.markdown('<div class="main-header">🧠 NeuroPass</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Blood-Brain Barrier Penetration Prediction</div>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.caption(f"Build: {APP_BUILD}")
        st.markdown("""
        This application predicts whether a drug molecule can cross the **Blood-Brain Barrier (BBB)**
        using a neural network trained on molecular fingerprints and physicochemical descriptors.

        **Model Performance:**
        - 85.1% Accuracy
        - 93.2% Precision
        - 0.90 ROC-AUC

        **How to use:**
        1. Enter a SMILES string or select an example
        2. Click "Predict"
        3. View the prediction and molecular properties
        """)

        st.header("Example Molecules")
        examples = {
            "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Dopamine": "C1=CC(=C(C=C1CCN)O)O",
            "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "Nicotine": "CN1CCCC1C2=CN=CC=C2"
        }

        selected_example = st.selectbox(
            "Select an example:", [""] + list(examples.keys()))

        if selected_example:
            st.session_state.example_smiles = examples[selected_example]

    # Load model
    model, preprocessor, device, config = load_model_and_preprocessor()

    if model is None:
        st.error("Failed to load model. Please check the model files.")
        return

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Molecule")

        # SMILES input
        default_smiles = st.session_state.get('example_smiles', '')
        smiles_input = st.text_input(
            "Enter SMILES string:",
            value=default_smiles,
            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O",
            help="Simplified Molecular Input Line Entry System (SMILES) notation"
        )

        predict_button = st.button(
            "🔬 Predict BBB Penetration", type="primary", use_container_width=True)

        if predict_button and smiles_input:
            with st.spinner("Analyzing molecule..."):
                # Make prediction
                result = predict_molecule(
                    smiles_input, model, preprocessor, device)

                if result['valid']:
                    st.session_state.prediction_result = result
                    st.session_state.current_smiles = smiles_input
                else:
                    st.error(f"❌ {result['error']}")
                    return

        # Display molecular structure
        if smiles_input:
            st.subheader("Molecular Structure")
            img = draw_molecule(smiles_input)
            if img:
                st.image(img, use_container_width=True)
            else:
                st.info(
                    "Molecular visualization unavailable (drawing backend not available)")

    with col2:
        st.subheader("Prediction Results")

        if 'prediction_result' in st.session_state and st.session_state.get('current_smiles') == smiles_input:
            result = st.session_state.prediction_result

            # Display prediction
            if result['prediction'] == 1:
                st.markdown(
                    f'<div class="prediction-positive">✅ {result["prediction_label"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="prediction-negative">❌ {result["prediction_label"]}</div>',
                    unsafe_allow_html=True
                )

            # Confidence metrics
            st.subheader("Confidence Scores")
            col_a, col_b = st.columns(2)

            with col_a:
                st.metric(
                    "Model Confidence",
                    f"{result['confidence']:.1%}",
                    help="Confidence in the predicted class"
                )
                st.caption(result.get('confidence_band', ''))

            with col_b:
                st.metric(
                    "BBB Penetration",
                    f"{result['probability_positive']:.1%}",
                    help="Probability of BBB penetration"
                )

            # Probability breakdown
            st.subheader("Probability Breakdown")
            prob_data = pd.DataFrame({
                'Class': ['Does not penetrate', 'Penetrates BBB'],
                'Probability': [result['probability_negative'], result['probability_positive']]
            })

            st.bar_chart(prob_data.set_index('Class'))

            # Molecular properties
            st.subheader("Molecular Properties")
            properties = get_molecular_properties(smiles_input)

            if properties:
                prop_df = pd.DataFrame(list(properties.items()), columns=[
                                       'Property', 'Value'])
                st.table(prop_df)
            else:
                st.info("Molecular properties unavailable")

        else:
            st.info("👈 Enter a SMILES string and click 'Predict' to see results")


    st.markdown("---")
    st.subheader("Compare molecules")
    st.caption("Run the same trained model across a shortlist. Scores are model estimates, not measured permeability.")
    threshold = st.slider("BBB-positive decision threshold", 0.05, 0.95, 0.50, 0.05)
    batch_text = st.text_area("One SMILES per line", value="CC(=O)OC1=CC=CC=C1C(=O)O\nCN1CCCC1C2=CN=CC=C2", height=120)
    upload = st.file_uploader("Or upload a CSV with a smiles column (optional name column)", type=["csv"])
    source = pd.DataFrame({"smiles": batch_text.splitlines()})
    if upload is not None:
        try:
            source = pd.read_csv(upload, dtype=str).fillna("")
            if "smiles" not in source.columns:
                st.error("The CSV needs a column named smiles.")
                source = pd.DataFrame(columns=["smiles"])
        except Exception:
            st.error("The CSV could not be read.")
            source = pd.DataFrame(columns=["smiles"])
    if st.button("Compare shortlist", type="primary"):
        if len(source) > 200:
            st.error("Upload at most 200 molecules per batch.")
        else:
            rows = []
            with st.spinner("Scoring shortlist..."):
                for index, row in source.iterrows():
                    smi = str(row["smiles"]).strip()
                    if not smi:
                        continue
                    record = {"Name": str(row.get("name", "")) or f"Molecule {index + 1}", "SMILES": smi}
                    try:
                        prediction = predict_molecule(smi, model, preprocessor, device)
                        if not prediction["valid"]:
                            record["Status"] = prediction["error"]
                        else:
                            record.update({"Status": "Scored", "BBB-positive score": prediction["probability_positive"]})
                            mol = Chem.MolFromSmiles(smi)
                            record.update({"MW (g/mol)": Descriptors.MolWt(mol), "LogP": Descriptors.MolLogP(mol),
                                           "TPSA (Å²)": Descriptors.TPSA(mol),
                                           "H-bond donors": Descriptors.NumHDonors(mol),
                                           "H-bond acceptors": Descriptors.NumHAcceptors(mol)})
                    except Exception:
                        record["Status"] = "Unable to score this molecule"
                    rows.append(record)
            st.session_state.shortlist = rows
    if st.session_state.get("shortlist"):
        comparison = pd.DataFrame(st.session_state.shortlist)
        if "BBB-positive score" in comparison:
            comparison["Decision"] = comparison["BBB-positive score"].apply(
                lambda score: "Invalid input" if pd.isna(score) else ("BBB+" if score >= threshold else "BBB−"))
            scored = comparison.dropna(subset=["BBB-positive score"])
            st.bar_chart(scored.set_index("Name")["BBB-positive score"])
            st.caption(f"Decision threshold: {threshold:.0%}. Moving the threshold changes decisions, not model scores.")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.download_button("Download comparison CSV", comparison.to_csv(index=False).encode("utf-8"),
                           "neuropass-comparison.csv", "text/csv")

    with st.expander("Inspect model evaluation and mistakes"):
        report_path = Path("experiments/benchmark/summary.json")
        if report_path.exists():
            report = json.loads(report_path.read_text())
            metrics = pd.DataFrame(report["results"]).T
            st.dataframe(metrics.drop(columns=["confusion_matrix"]), use_container_width=True)
            st.caption(f"Saved test split: {report['valid_test_rows']} valid molecules. Exact canonical structures shared with training: {report['canonical_structure_overlap_train_test']}.")
            st.write("The logistic baseline has higher accuracy on this split; the neural network has higher ROC-AUC. A stronger model claim needs scaffold-disjoint validation.")
            st.caption("These are retrospective results. Model scores are uncalibrated and are not clinical evidence.")
            errors_path = Path("experiments/benchmark/errors.csv")
            if errors_path.exists():
                errors = pd.read_csv(errors_path)
                st.write("Incorrect test predictions")
                st.dataframe(errors[["name", "smiles", "p_np", "bbb_score", "predicted_label"]], hide_index=True)
                st.download_button("Download test errors", errors.to_csv(index=False), "neuropass-test-errors.csv", "text/csv")

    # Information section
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <strong>ℹ️ About Blood-Brain Barrier Prediction</strong><br>
    The blood-brain barrier (BBB) is a selective membrane that protects the brain from harmful substances
    while allowing essential nutrients to pass through. Predicting BBB penetration is crucial for:
    <ul>
    <li>Central nervous system (CNS) drug development</li>
    <li>Reducing costly laboratory experiments</li>
    <li>Accelerating pharmaceutical research</li>
    </ul>
    This model uses Morgan fingerprints and physicochemical descriptors to predict BBB penetration
    with 93.2% precision, helping researchers identify promising drug candidates early in the development process.
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666;"><strong>NeuroPass</strong> - Built with PyTorch, RDKit, and Streamlit | '
        'Chemistry + AI Integration</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
