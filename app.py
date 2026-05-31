"""
Streamlit Web Interface for Blood-Brain Barrier Penetration Prediction
"""

import os
import sys
import streamlit as st
import torch
import pandas as pd
from PIL import Image
from io import BytesIO
import base64

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data import MoleculePreprocessor
from src.models import create_model
from src.utils import load_config

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    st.warning("RDKit not available. Molecular visualization will be disabled.")


# Page configuration
st.set_page_config(
    page_title="BBB Penetration Predictor",
    page_icon="🧬",
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
        # Load configuration
        config = load_config('config/config.yaml')

        # Load preprocessor
        preprocessor = MoleculePreprocessor(config)
        scaler_path = os.path.join(config['data']['processed_dir'], 'scaler.pkl')

        try:
            preprocessor.load_scaler(scaler_path)
        except FileNotFoundError:
            st.warning("Scaler not found. Predictions will use non-scaled features.")

        # Load model
        input_size = preprocessor.get_feature_dim()
        model = create_model(config, input_size)

        device = config['training']['device']
        model_path = 'models/checkpoints/best_model.pth'

        checkpoint = torch.load(model_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

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

    # Convert to tensor
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)

    return {
        'valid': True,
        'prediction': int(prediction.item()),
        'prediction_label': 'Penetrates BBB' if prediction.item() == 1 else 'Does not penetrate BBB',
        'confidence': float(probabilities[0, prediction.item()].item()),
        'probability_negative': float(probabilities[0, 0].item()),
        'probability_positive': float(probabilities[0, 1].item())
    }


def draw_molecule(smiles: str):
    """Draw molecule structure from SMILES."""
    if not RDKIT_AVAILABLE:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        img = Draw.MolToImage(mol, size=(400, 400))
        return img
    except Exception as e:
        st.error(f"Error drawing molecule: {str(e)}")
        return None


def get_molecular_properties(smiles: str):
    """Calculate basic molecular properties."""
    if not RDKIT_AVAILABLE:
        return {}

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        properties = {
            'Molecular Weight': f"{Descriptors.MolWt(mol):.2f} g/mol",
            'LogP': f"{Descriptors.MolLogP(mol):.2f}",
            'H-Bond Donors': Descriptors.NumHDonors(mol),
            'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),
            'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
            'Aromatic Rings': Descriptors.NumAromaticRings(mol),
            'TPSA': f"{Descriptors.TPSA(mol):.2f} Ų"
        }
        return properties
    except Exception as e:
        return {}


def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<div class="main-header">🧬 Blood-Brain Barrier Penetration Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered molecular property prediction using neural networks</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("About")
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

        selected_example = st.selectbox("Select an example:", [""] + list(examples.keys()))

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

        predict_button = st.button("🔬 Predict BBB Penetration", type="primary", use_container_width=True)

        if predict_button and smiles_input:
            with st.spinner("Analyzing molecule..."):
                # Make prediction
                result = predict_molecule(smiles_input, model, preprocessor, device)

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
                st.info("Molecular visualization unavailable")

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
                prop_df = pd.DataFrame(list(properties.items()), columns=['Property', 'Value'])
                st.table(prop_df)
            else:
                st.info("Molecular properties unavailable")

        else:
            st.info("👈 Enter a SMILES string and click 'Predict' to see results")

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
        '<div style="text-align: center; color: #666;">Built with PyTorch, RDKit, and Streamlit | '
        'Chemistry + AI Integration</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
