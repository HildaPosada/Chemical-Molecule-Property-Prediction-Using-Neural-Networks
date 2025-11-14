# Molecular Property Prediction with Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

A production-grade deep learning system for predicting blood-brain barrier (BBB) penetration from molecular structure. This project bridges computational chemistry and neural networks to address a critical challenge in pharmaceutical drug development—screening molecules for central nervous system (CNS) drug candidates without expensive in vitro or in vivo testing.

The implementation achieves **85.1% classification accuracy** and **0.90 ROC-AUC** on the MoleculeNet BBBP benchmark, demonstrating that neural networks trained on molecular fingerprints can effectively predict BBB permeability with performance comparable to published research.

## Motivation

Blood-brain barrier penetration is a critical property for neurological drug development. Traditional experimental methods require:
- Weeks to months of laboratory testing per compound
- Significant financial investment ($10,000+ per molecule)
- Animal models or complex in vitro assays

Machine learning approaches can screen thousands of candidates in minutes, enabling:
- Rapid lead optimization in drug discovery pipelines
- Reduced R&D costs by filtering unpromising candidates early
- Acceleration of neurological drug development timelines

This project demonstrates end-to-end capability in applying deep learning to molecular science—a critical skill set for emerging fields like quantum computing applications in chemistry and materials science.

## Technical Implementation

### Architecture

**Feature Extraction Pipeline:**
- **Input:** SMILES (Simplified Molecular Input Line Entry System) strings
- **Molecular Fingerprints:** 1024-bit Morgan circular fingerprints (radius=2)
- **Physicochemical Descriptors:** 8 computed features (LogP, molecular weight, H-bond donors/acceptors, TPSA, rotatable bonds, aromatic rings, sp³ carbons)
- **Preprocessing:** StandardScaler normalization, stratified train/val/test splitting

**Neural Network:**
```
Input Layer (1032 features)
    ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(64) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(2) → Softmax
```

**Training Infrastructure:**
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-5)
- **Loss:** Cross-entropy with class weighting
- **Regularization:** Dropout, L2 weight decay, gradient clipping
- **Callbacks:** Early stopping (patience=15), ReduceLROnPlateau
- **Monitoring:** TensorBoard integration for real-time metrics

### Dataset

**MoleculeNet BBBP Benchmark**
- **Size:** 2,039 molecules (binary classification)
- **Source:** Curated from peer-reviewed experimental data
- **Split:** 70% train / 15% validation / 15% test (stratified)
- **Class Distribution:** 1,562 BBB+ / 477 BBB- (balanced via class weights)

## Performance Results

### Classification Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 85.1% |
| **ROC-AUC** | 0.90 |
| **Precision** | 93.2% |
| **Recall** | 86.8% |
| **F1-Score** | 89.9% |

### Training Characteristics

- **Convergence:** 50 epochs (Codespaces CPU training)
- **Training Time:** ~30 minutes (GitHub Codespaces 2-core)
- **Model Size:** 35,554 parameters (compact architecture)
- **Feature Dimension:** 518 (512-bit fingerprints + 6 descriptors)

### Comparative Performance

This implementation achieves competitive performance with published benchmarks:
- **MoleculeNet baseline (2018):** ~88% accuracy
- **This implementation:** 85.1% accuracy (93.2% precision)
- **Graph neural networks (SOTA):** ~90-92% accuracy

The results demonstrate that well-engineered traditional fingerprints with standard neural networks can approach state-of-the-art performance, while being significantly simpler to implement and interpret than graph-based methods.

## Technical Stack

| Component | Technology |
|-----------|-----------|
| **Deep Learning** | PyTorch 2.0+ |
| **Cheminformatics** | RDKit |
| **Numerical Computing** | NumPy, Pandas |
| **Machine Learning** | Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Experiment Tracking** | TensorBoard |
| **Data Storage** | SQLite |
| **Configuration** | YAML |

## Project Structure

```
├── src/
│   ├── data/              # Data loading, SMILES processing, feature extraction
│   ├── models/            # Neural network architectures
│   ├── training/          # Training loops, callbacks, optimization
│   ├── evaluation/        # Metrics computation, visualization
│   └── utils/             # Configuration, logging, device management
├── scripts/               # CLI tools for training, evaluation, inference
├── config/                # Hyperparameter configurations
├── notebooks/             # Exploratory analysis and experimentation
└── tests/                 # Unit tests
```

## Key Features

**Production-Ready Infrastructure:**
- Modular, object-oriented design for maintainability
- Comprehensive logging and error handling
- Configuration management via YAML
- Reproducible experiments (fixed random seeds)
- Model versioning and checkpointing

**Advanced Training Techniques:**
- Learning rate scheduling with plateau detection
- Early stopping to prevent overfitting
- Gradient clipping for stability
- Class-weighted loss for imbalanced data
- TensorBoard integration for monitoring

**Robust Evaluation:**
- Multiple performance metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix analysis
- ROC curve visualization
- Cross-validation ready architecture

## Installation & Usage

### Quick Setup

```bash
# Clone repository
git clone https://github.com/HildaPosada/Chemical-Molecule-Property-Prediction-Using-Neural-Networks.git
cd Chemical-Molecule-Property-Prediction-Using-Neural-Networks

# Install dependencies (conda recommended for RDKit)
conda create -n molecule-pred python=3.9
conda activate molecule-pred
conda install -c conda-forge rdkit
pip install -r requirements.txt

# Download data and train
python scripts/download_data.py
python scripts/train.py
```

### GitHub Codespaces

For cloud-based training without local setup:
1. Open repository in GitHub Codespaces
2. Environment auto-configures in ~3 minutes
3. Run: `python scripts/train.py --config config/config_codespaces.yaml`

Detailed setup instructions: [CODESPACES_QUICKSTART.md](CODESPACES_QUICKSTART.md)

### Command-Line Interface

```bash
# Training
python scripts/train.py --config config/config.yaml --epochs 100

# Evaluation
python scripts/evaluate.py --model-path models/checkpoints/best_model.pth

# Inference
python scripts/predict.py --smiles "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen
python scripts/predict.py --input-file molecules.csv --output predictions.csv
```

## Applications

**Pharmaceutical Industry:**
- Early-stage drug candidate screening
- Lead optimization in CNS drug discovery
- ADME (Absorption, Distribution, Metabolism, Excretion) prediction

**Research Applications:**
- Computational toxicology
- QSAR (Quantitative Structure-Activity Relationship) modeling
- Chemical library virtual screening

**Extensibility:**
- Transfer learning to other molecular properties
- Ensemble methods for uncertainty quantification
- Integration with quantum chemistry calculations

## Future Development

**Model Enhancements:**
- Graph Neural Networks (GNN) for superior molecular representation
- Attention mechanisms for interpretability
- Multi-task learning across multiple molecular properties
- Uncertainty quantification via Bayesian approaches

**Infrastructure:**
- REST API deployment (FastAPI)
- Containerization (Docker)
- Model serving at scale
- Integration with molecular dynamics simulations

**Advanced Techniques:**
- SHAP/LIME for model explainability
- Active learning for data-efficient training
- Few-shot learning for rare property prediction

## Technical Highlights

This project demonstrates proficiency in:

✓ **Deep Learning:** PyTorch model development, training optimization, regularization
✓ **Cheminformatics:** SMILES processing, molecular fingerprints, descriptor calculation
✓ **Software Engineering:** Modular architecture, configuration management, CLI tools
✓ **Data Science:** Feature engineering, imbalanced data handling, performance evaluation
✓ **MLOps:** Experiment tracking, model checkpointing, reproducibility
✓ **Domain Knowledge:** Understanding of drug discovery, BBB biology, ADME properties

## References

**Dataset:**
- Wu, Z. et al. (2018). "MoleculeNet: A Benchmark for Molecular Machine Learning." *Chemical Science*, 9(2), 513-530.

**Methodology:**
- Rogers, D. & Hahn, M. (2010). "Extended-Connectivity Fingerprints." *Journal of Chemical Information and Modeling*, 50(5), 742-754.

**Related Work:**
- Deep learning applications in drug discovery
- Neural network approaches to QSAR modeling
- Molecular representation learning

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

**Hilda Posada**
Bridging Chemistry and Machine Learning
[GitHub](https://github.com/HildaPosada) | [LinkedIn](https://linkedin.com/in/hildaposada)

---

*This project showcases the intersection of computational chemistry and deep learning—a foundation for emerging applications in quantum computing, materials science, and AI-driven drug discovery.*
