"""Quick test script to verify setup is correct."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")

    tests = {
        "PyTorch": lambda: __import__("torch"),
        "NumPy": lambda: __import__("numpy"),
        "Pandas": lambda: __import__("pandas"),
        "Scikit-learn": lambda: __import__("sklearn"),
        "Matplotlib": lambda: __import__("matplotlib"),
        "Seaborn": lambda: __import__("seaborn"),
        "RDKit": lambda: __import__("rdkit"),
        "YAML": lambda: __import__("yaml"),
    }

    all_passed = True
    for name, import_func in tests.items():
        try:
            import_func()
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - {e}")
            all_passed = False

    return all_passed


def test_modules():
    """Test that project modules can be imported."""
    print("\nTesting project modules...")

    modules = [
        "src.data",
        "src.models",
        "src.training",
        "src.evaluation",
        "src.utils",
    ]

    all_passed = True
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module} - {e}")
            all_passed = False

    return all_passed


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")

    try:
        from src.utils import load_config
        config = load_config("config/config_codespaces.yaml")
        print(f"  ✓ Configuration loaded")
        print(f"  ✓ Device: {config['training']['device']}")
        print(f"  ✓ Batch size: {config['training']['batch_size']}")
        print(f"  ✓ Epochs: {config['training']['num_epochs']}")
        return True
    except Exception as e:
        print(f"  ✗ Configuration loading failed - {e}")
        return False


def test_rdkit():
    """Test RDKit functionality."""
    print("\nTesting RDKit functionality...")

    try:
        from rdkit import Chem

        # Test SMILES parsing
        smiles = "CCO"  # Ethanol
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            print(f"  ✓ SMILES parsing works")
            print(f"  ✓ Ethanol (CCO) has {mol.GetNumAtoms()} atoms")
            return True
        else:
            print(f"  ✗ SMILES parsing failed")
            return False
    except Exception as e:
        print(f"  ✗ RDKit test failed - {e}")
        return False


def test_pytorch():
    """Test PyTorch functionality."""
    print("\nTesting PyTorch...")

    try:
        import torch

        # Create a simple tensor
        x = torch.randn(5, 3)
        print(f"  ✓ Tensor creation works")
        print(f"  ✓ PyTorch version: {torch.__version__}")
        print(f"  ✓ CPU available: True")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")

        # Test simple neural network
        model = torch.nn.Linear(3, 2)
        output = model(x)
        print(f"  ✓ Neural network forward pass works")

        return True
    except Exception as e:
        print(f"  ✗ PyTorch test failed - {e}")
        return False


def test_directories():
    """Test that required directories exist."""
    print("\nTesting directory structure...")

    required_dirs = [
        "data/raw",
        "data/processed",
        "models/saved_models",
        "models/checkpoints",
        "results/figures",
        "results/metrics",
        "logs",
        "config",
        "scripts",
        "src",
    ]

    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - missing")
            all_exist = False

    return all_exist


def main():
    """Run all tests."""
    print("="*60)
    print("Molecular Property Prediction - Setup Verification")
    print("="*60)
    print()

    results = {
        "Package imports": test_imports(),
        "Project modules": test_modules(),
        "Configuration": test_config(),
        "RDKit functionality": test_rdkit(),
        "PyTorch functionality": test_pytorch(),
        "Directory structure": test_directories(),
    }

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    print()

    if all(results.values()):
        print("🎉 All tests passed! Your environment is ready.")
        print()
        print("Next steps:")
        print("1. Download data: python scripts/download_data.py")
        print("2. Train model: python scripts/train.py --config config/config_codespaces.yaml")
        print("3. Start TensorBoard: tensorboard --logdir=runs --bind_all")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
