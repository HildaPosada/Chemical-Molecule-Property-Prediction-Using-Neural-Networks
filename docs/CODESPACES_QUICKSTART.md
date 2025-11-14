# 🚀 GitHub Codespaces Quick Start

**Perfect for training without local setup!**

## Step 1: Launch Codespace (1 minute)

1. Go to your GitHub repository
2. Click green **"Code"** button
3. Click **"Codespaces"** tab
4. Click **"Create codespace on claude/molecule-property-prediction-01TJQFmvbwnP2sfFkujXVtTY"**

The environment will automatically install everything (takes ~2-3 minutes).

## Step 2: Verify Setup (30 seconds)

Once the terminal shows "✅ Setup complete!", run:

```bash
python scripts/test_setup.py
```

You should see all tests pass ✓

## Step 3: Train Your Model (~30 minutes)

```bash
# Download the dataset (~2,000 molecules)
python scripts/download_data.py

# Train the model (Codespaces-optimized config)
python scripts/train.py --config config/config_codespaces.yaml
```

**What to expect:**
- Training time: 20-30 minutes on Codespaces CPU
- You'll see progress after each epoch
- Model checkpoints saved automatically to `models/checkpoints/`

### Monitor Training (Optional)

In a new terminal:

```bash
tensorboard --logdir=runs --bind_all
```

VS Code will show a popup - click it to open TensorBoard in your browser!

## Step 4: Evaluate Results (2 minutes)

```bash
# Evaluate the trained model
python scripts/evaluate.py \
  --model-path models/checkpoints/best_model.pth \
  --config config/config_codespaces.yaml

# Check the results
ls results/figures/        # Plots and visualizations
ls results/metrics/        # Performance metrics
```

## Step 5: Make Predictions

### Single molecule:

```bash
python scripts/predict.py \
  --smiles "CC(C)Cc1ccc(cc1)C(C)C(O)=O" \
  --config config/config_codespaces.yaml
```

### Batch predictions:

```bash
# Create a test file
echo "smiles
CCO
CC(=O)O
CN1C=NC2=C1C(=O)N(C(=O)N2C)C" > test_molecules.csv

# Predict
python scripts/predict.py \
  --input-file test_molecules.csv \
  --output predictions.csv \
  --config config/config_codespaces.yaml

# View results
cat predictions.csv
```

## 📊 View Your Results

After training, you'll have:

1. **Training curves**: `results/figures/training_history.png`
2. **Confusion matrix**: `results/figures/test_confusion_matrix.png`
3. **ROC curve**: `results/figures/test_roc_curve.png`
4. **Metrics**: `results/metrics/test_metrics.json`

Right-click any file → **Download** to save to your computer!

## ⚡ Quick Commands Cheat Sheet

```bash
# Download data
python scripts/download_data.py

# Train (full)
python scripts/train.py --config config/config_codespaces.yaml

# Train (quick test - 20 epochs, 500 molecules)
python scripts/train.py \
  --config config/config_codespaces.yaml \
  --epochs 20

# Evaluate
python scripts/evaluate.py \
  --model-path models/checkpoints/best_model.pth \
  --config config/config_codespaces.yaml

# Predict single molecule
python scripts/predict.py \
  --smiles "YOUR_SMILES_HERE" \
  --config config/config_codespaces.yaml

# Start TensorBoard
tensorboard --logdir=runs --bind_all

# Test setup
python scripts/test_setup.py

# Check logs
tail -f logs/*.log
```

## 💾 Save Your Work

**Important**: Codespaces can timeout if inactive!

### Commit & Push Regularly

```bash
git add models/ results/ logs/
git commit -m "Training results - achieved X% accuracy"
git push
```

### Download Important Files

Right-click in VS Code → Download:
- `models/checkpoints/best_model.pth`
- `results/figures/` folder
- `results/metrics/` folder

## 🐛 Troubleshooting

### "RDKit not found"

```bash
pip install rdkit
```

### Training too slow?

Use smaller dataset for testing:

```bash
# Edit config/config_codespaces.yaml
# Add: max_molecules: 500

python scripts/train.py --config config/config_codespaces.yaml --epochs 20
```

### Out of memory?

```bash
# Smaller batch size
python scripts/train.py --config config/config_codespaces.yaml --batch-size 32
```

### Can't see TensorBoard?

1. Click **"Ports"** tab in VS Code (bottom panel)
2. Find port **6006**
3. Click the 🌐 globe icon

## 📈 Expected Results

With the Codespaces config, you should get:

- **Accuracy**: 85-88%
- **ROC-AUC**: 0.88-0.91
- **Training time**: 20-30 minutes
- **Model size**: ~1 MB

## 💡 Tips for Success

1. **Start with a quick run first**:
   ```bash
   # 10 epochs, see if everything works
   python scripts/train.py --config config/config_codespaces.yaml --epochs 10
   ```

2. **Monitor resource usage**:
   - Click gear icon (⚙️) in VS Code
   - Check "Machine Type" to see your resources

3. **Use free tier wisely**:
   - You get 120 core-hours/month free
   - 2-core machine = 60 hours of usage
   - Stop Codespace when not using: Settings → Stop Codespace

4. **Experiment with hyperparameters**:
   ```bash
   # Try different learning rates
   python scripts/train.py --config config/config_codespaces.yaml --lr 0.0001
   python scripts/train.py --config config/config_codespaces.yaml --lr 0.01
   ```

## 🎯 For Your IBM Quantum Portfolio

After training, showcase:

1. **Performance metrics** from `results/metrics/`
2. **Visualizations** from `results/figures/`
3. **Model architecture** (in README)
4. **Training curves** showing convergence
5. **Example predictions** on drug-like molecules

## 📚 Full Documentation

- Detailed README: `README.md`
- Codespaces guide: `.devcontainer/README.md`
- Contributing: `CONTRIBUTING.md`

## ❓ Questions?

- Check logs: `logs/` folder
- Re-run setup: `bash .devcontainer/setup.sh`
- Test environment: `python scripts/test_setup.py`

---

**Ready to train?** Just run:

```bash
python scripts/download_data.py && \
python scripts/train.py --config config/config_codespaces.yaml
```

Then sit back and watch your model learn! 🧠✨
