# Molecular Property Prediction with Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A production-grade deep learning system for predicting blood-brain barrier penetration from molecular structure. Achieved 85.1% accuracy with 93.2% precision on the MoleculeNet BBBP benchmark.**

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Problem** | Predict whether drug molecules can cross the blood-brain barrier (critical for CNS drugs) |
| **Approach** | Neural network trained on molecular fingerprints (RDKit) + physicochemical descriptors |
| **Results** | 85.1% accuracy, 0.90 ROC-AUC, **93.2% precision** |
| **Tech Stack** | PyTorch, RDKit, Scikit-learn, TensorBoard |
| **Impact** | Reduces expensive laboratory screening; demonstrates ML + chemistry integration |

## Why This Matters

Blood-brain barrier (BBB) penetration prediction accelerates neurological drug development by:
- **Reducing costs:** Eliminates need for expensive in vitro/in vivo testing ($10,000+ per compound)
- **Accelerating timelines:** Screen thousands of candidates in minutes vs. weeks
- **Improving success rates:** High precision (93.2%) minimizes false positives

**Relevance to Quantum Computing:** Molecular property prediction is foundational to quantum chemistry simulations—understanding molecular behavior is essential for quantum computing applications in materials science and drug discovery.

---

## Performance Results

### Classification Metrics

| Metric | Score | Why It Matters |
|--------|-------|----------------|
| **Accuracy** | 85.1% | Overall correctness |
| **Precision** | **93.2%** | Low false positives (critical for drug screening) |
| **Recall** | 86.8% | Captures most viable candidates |
| **F1-Score** | 89.9% | Balanced performance |
| **ROC-AUC** | 0.90 | Strong discrimination ability |

### Training Details
- **Dataset:** 2,039 molecules from MoleculeNet BBBP benchmark
- **Training Time:** 30 minutes on GitHub Codespaces (2-core CPU)
- **Model Size:** 35,554 parameters (compact and efficient)
- **Convergence:** 50 epochs with early stopping capability

### Comparative Performance
```
MoleculeNet Baseline (2018):  ~88% accuracy
This Implementation:          85.1% accuracy (93.2% precision)
Graph Neural Networks (SOTA): ~90-92% accuracy
```

Results demonstrate that engineered molecular fingerprints with standard neural networks achieve competitive performance while remaining interpretable and computationally efficient.

---

## Visualizations

<table>
<tr>
<td width="33%">

**Confusion Matrix**
![Confusion Matrix](results/figures/test_confusion_matrix.png)
Strong true positive rate (204) with minimal false positives (15)

</td>
<td width="33%">

**ROC Curve**
![ROC Curve](results/figures/test_roc_curve.png)
AUC = 0.90 shows excellent discrimination

</td>
<td width="33%">

**Performance Metrics**
![Metrics](results/figures/test_metrics.png)
Comprehensive evaluation across all metrics

</td>
</tr>
</table>

---

## Technical Architecture

### Feature Pipeline
```
SMILES String → RDKit Processing → Feature Engineering → Neural Network → Prediction
                                    ↓
                    • 512-bit Morgan fingerprints (molecular structure)
                    • 6 physicochemical descriptors (LogP, MW, H-bonds, TPSA, etc.)
                    • StandardScaler normalization
```

### Neural Network Architecture
```
Input (518 features)
    ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(64) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(2) → Softmax → Prediction (BBB+/BBB-)
```

### Training Strategy
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-5)
- **Loss:** Cross-entropy with class weighting (handles imbalanced data)
- **Regularization:** Dropout + L2 weight decay + gradient clipping
- **Callbacks:** Early stopping, ReduceLROnPlateau, model checkpointing
- **Monitoring:** TensorBoard integration

---

## Project Structure

```
├── src/                    # Core implementation
│   ├── data/              # Data loading, SMILES processing, feature extraction
│   ├── models/            # Neural network architectures
│   ├── training/          # Training loops, callbacks, optimization
│   ├── evaluation/        # Metrics computation, visualization
│   └── utils/             # Configuration, logging, device management
├── scripts/               # Command-line tools (train, evaluate, predict)
├── config/                # YAML configurations
├── results/               # Visualizations and metrics
├── models/                # Trained model checkpoints
└── notebooks/             # Exploratory analysis
```

---

## Quick Start

### Option 1: GitHub Codespaces (Recommended)
```bash
# Open in Codespaces → auto-setup in ~3 minutes
python scripts/download_data.py
python scripts/train.py --config config/config_codespaces.yaml
```

### Option 2: Local Installation
```bash
# Clone and setup
git clone https://github.com/HildaPosada/Chemical-Molecule-Property-Prediction-Using-Neural-Networks.git
cd Chemical-Molecule-Property-Prediction-Using-Neural-Networks

# Install dependencies
conda create -n molecule-pred python=3.9
conda activate molecule-pred
conda install -c conda-forge rdkit
pip install -r requirements.txt

# Train model
python scripts/download_data.py
python scripts/train.py
```

### Command-Line Interface
```bash
# Training
python scripts/train.py --config config/config.yaml --epochs 100

# Evaluation
python scripts/evaluate.py --model-path models/checkpoints/best_model.pth

# Prediction
python scripts/predict.py --smiles "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen
```

---

## Key Technical Highlights

**Deep Learning Expertise:**
- PyTorch model development with custom architectures
- Advanced training techniques (LR scheduling, early stopping, gradient clipping)
- Production-ready MLOps practices (checkpointing, logging, reproducibility)

**Cheminformatics Knowledge:**
- SMILES processing and validation
- Molecular fingerprint generation (RDKit Morgan fingerprints)
- Physicochemical descriptor calculation
- Understanding of drug ADME properties

**Software Engineering:**
- Modular, object-oriented architecture
- Configuration management (YAML)
- Comprehensive testing and error handling
- CLI tools for all operations

**Data Science:**
- Handling imbalanced datasets (class weighting)
- Feature engineering from molecular structures
- Stratified train/val/test splitting
- Multiple evaluation metrics

---

## Applications

**Pharmaceutical Industry:**
- Early-stage drug candidate screening
- Lead optimization in CNS drug discovery
- Virtual screening of chemical libraries

**Research:**
- Computational toxicology
- QSAR modeling
- Integration with quantum chemistry simulations

---

## References

- **Wu, Z. et al. (2018).** "MoleculeNet: A Benchmark for Molecular Machine Learning." *Chemical Science*, 9(2), 513-530.
- **Rogers, D. & Hahn, M. (2010).** "Extended-Connectivity Fingerprints." *J. Chem. Inf. Model.*, 50(5), 742-754.

## License

MIT License - see [LICENSE](LICENSE)

## Contact

**Hilda Posada**
Bridging Chemistry and Machine Learning
[GitHub](https://github.com/HildaPosada) | [LinkedIn](https://linkedin.com/in/hildaposada)

---

*This project demonstrates the intersection of computational chemistry and deep learning—foundational skills for quantum computing applications in molecular science and materials discovery.*
