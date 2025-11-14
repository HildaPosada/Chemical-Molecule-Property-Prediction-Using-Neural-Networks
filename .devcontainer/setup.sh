#!/bin/bash

# GitHub Codespaces Setup Script for Molecule Property Prediction
# This script runs automatically when the Codespace is created

set -e

echo "================================================"
echo "Setting up Molecular Property Prediction Project"
echo "================================================"

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update -qq

# Install system dependencies for RDKit
echo "🔬 Installing system dependencies for RDKit..."
sudo apt-get install -y -qq \
    libboost-all-dev \
    libcairo2-dev \
    libeigen3-dev \
    build-essential

# Upgrade pip
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel -q

# Install RDKit via pip (works in Linux)
echo "🧪 Installing RDKit..."
pip install rdkit -q

# Install other Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt -q

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/raw data/processed
mkdir -p models/saved_models models/checkpoints
mkdir -p results/figures results/metrics
mkdir -p logs runs

# Install the package in development mode
echo "📦 Installing package in development mode..."
pip install -e . -q

echo ""
echo "✅ Setup complete!"
echo ""
echo "================================================"
echo "Next steps:"
echo "================================================"
echo "1. Download data:"
echo "   python scripts/download_data.py"
echo ""
echo "2. Train model (CPU optimized):"
echo "   python scripts/train.py"
echo ""
echo "3. Monitor training with TensorBoard:"
echo "   tensorboard --logdir=runs --bind_all"
echo "   (then click the port 6006 link in VS Code)"
echo ""
echo "4. Evaluate model:"
echo "   python scripts/evaluate.py"
echo "================================================"
echo ""
echo "⚡ Quick test:"
echo "   python scripts/test_setup.py"
echo ""
