# GitHub Codespaces Setup

This directory contains the configuration for running this project in GitHub Codespaces.

## What Happens Automatically

When you open this repository in Codespaces, the following happens automatically:

1. **Container Setup**: A Python 3.9 development environment is created
2. **Dependencies**: All required packages are installed (including RDKit)
3. **VS Code Extensions**: Useful extensions for Python development are installed
4. **Directory Structure**: All necessary folders are created
5. **Port Forwarding**: Ports 6006 (TensorBoard) and 8888 (Jupyter) are forwarded

## Getting Started in Codespaces

### 1. Create a Codespace

- Go to your GitHub repository
- Click the green "Code" button
- Select "Codespaces" tab
- Click "Create codespace on main" (or your branch)

### 2. Wait for Setup

The setup script will run automatically. You'll see output in the terminal showing:
- Package installation
- Directory creation
- Setup completion

This takes about 2-3 minutes.

### 3. Verify Setup

Run the test script to ensure everything is working:

```bash
python scripts/test_setup.py
```

You should see all tests pass with ✓ marks.

## Training the Model in Codespaces

### Quick Start

```bash
# 1. Download dataset
python scripts/download_data.py

# 2. Train model (using Codespaces-optimized config)
python scripts/train.py --config config/config_codespaces.yaml

# 3. Start TensorBoard to monitor training
tensorboard --logdir=runs --bind_all
```

When TensorBoard starts, VS Code will show a notification. Click it to open TensorBoard in your browser.

### Configuration for Codespaces

We've created a special config file (`config/config_codespaces.yaml`) optimized for CPU training:

- **Smaller model**: 64→32 hidden units (vs 128→64)
- **Fewer epochs**: 50 (vs 100) for faster training
- **Larger batches**: 64 samples
- **Reduced features**: 512-bit fingerprints (vs 1024)
- **CPU device**: Forced to use CPU

**Expected training time**: ~20-30 minutes

### Monitoring Training

#### Option 1: TensorBoard (Recommended)

```bash
tensorboard --logdir=runs --bind_all
```

Then click the port 6006 link in VS Code's "Ports" panel.

#### Option 2: Log Files

```bash
# Watch training logs in real-time
tail -f logs/molecule_prediction_*.log
```

#### Option 3: VS Code Terminal

Training progress is printed directly to the terminal.

## Evaluating the Model

After training completes:

```bash
# Evaluate on test set
python scripts/evaluate.py --model-path models/checkpoints/best_model.pth --config config/config_codespaces.yaml

# View results
ls results/figures/
ls results/metrics/
```

## Making Predictions

```bash
# Single molecule prediction
python scripts/predict.py --smiles "CC(C)Cc1ccc(cc1)C(C)C(O)=O" --config config/config_codespaces.yaml

# Batch predictions
python scripts/predict.py --input-file molecules.csv --output predictions.csv --config config/config_codespaces.yaml
```

## Using Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Or use VS Code's built-in Jupyter support
# Just open notebooks/00_quickstart.ipynb
```

## Storage Considerations

Codespaces free tier provides:
- **Storage**: 15 GB (plenty for this project)
- **Compute**: 2 cores, 4 GB RAM
- **Hours**: 120 core-hours/month (60 hours on 2-core machine)

This project uses approximately:
- **Code**: ~10 MB
- **Data**: ~5 MB
- **Models**: ~1-2 MB per checkpoint
- **Logs/Results**: ~10-20 MB

**Total**: < 100 MB

## Performance Optimization Tips

### Speed Up Training

1. **Reduce dataset size** (for testing):
   ```yaml
   # In config_codespaces.yaml
   max_molecules: 500  # Use only 500 molecules
   ```

2. **Fewer epochs**:
   ```bash
   python scripts/train.py --config config/config_codespaces.yaml --epochs 20
   ```

3. **Larger batches**:
   ```bash
   python scripts/train.py --config config/config_codespaces.yaml --batch-size 128
   ```

### Save Compute Hours

- **Stop your Codespace** when not using it (Settings → Stop Codespace)
- **Commit frequently** - Codespaces can timeout
- **Download important results** before stopping

## Troubleshooting

### RDKit Import Error

If you see `ImportError: No module named 'rdkit'`:

```bash
pip install rdkit
```

### Out of Memory

If training crashes with OOM:

```bash
# Use smaller batch size
python scripts/train.py --config config/config_codespaces.yaml --batch-size 32

# Or reduce model size in config_codespaces.yaml:
# hidden_sizes: [32, 16]
```

### Slow Training

This is expected on CPU. To speed up:

1. Reduce dataset size (max_molecules: 500)
2. Use fewer epochs (--epochs 20)
3. Smaller model (hidden_sizes: [32, 16])

### Port Forwarding Issues

If TensorBoard doesn't open:

1. Click "Ports" tab in VS Code
2. Find port 6006
3. Click the globe icon to open in browser
4. Or use "Preview in Editor"

## Saving Your Work

### Commit Regularly

```bash
git add .
git commit -m "Training results after X epochs"
git push
```

### Download Results

Right-click on files/folders in VS Code → Download

Important to download:
- `models/checkpoints/best_model.pth`
- `results/figures/` (all plots)
- `results/metrics/` (performance metrics)

## Resources

- [Codespaces Documentation](https://docs.github.com/en/codespaces)
- [VS Code in Codespaces](https://code.visualstudio.com/docs/remote/codespaces)
- Project README: `../README.md`

## Questions?

Check the main README or open an issue on GitHub.
