# Contributing to Molecular Property Prediction

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/Chemical-Molecule-Property-Prediction-Using-Neural-Networks.git
   cd Chemical-Molecule-Property-Prediction-Using-Neural-Networks
   ```

3. **Set up your environment**:
   ```bash
   conda create -n molecule-pred python=3.9
   conda activate molecule-pred
   conda install -c conda-forge rdkit
   pip install -r requirements.txt
   ```

4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

Example:
```python
def calculate_molecular_weight(smiles: str) -> float:
    """
    Calculate molecular weight from SMILES string.

    Args:
        smiles: SMILES representation of molecule

    Returns:
        Molecular weight in g/mol
    """
    # Implementation here
    pass
```

### Code Quality

Before submitting, ensure your code passes:

```bash
# Format code
black src/ scripts/

# Check style
flake8 src/ scripts/

# Type checking
mypy src/

# Run tests
pytest tests/
```

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for good test coverage

```python
# Example test
def test_molecule_validation():
    """Test SMILES validation."""
    preprocessor = MoleculePreprocessor(config)
    assert preprocessor.validate_smiles("CCO") == True
    assert preprocessor.validate_smiles("invalid") == False
```

## Types of Contributions

### Bug Reports

When reporting bugs, include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages and stack traces

### Feature Requests

For feature requests:
- Describe the feature and its benefits
- Provide use cases
- Suggest possible implementation approach

### Code Contributions

Good areas to contribute:
- New molecular descriptors
- Additional model architectures
- Performance optimizations
- Documentation improvements
- Bug fixes

## Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Update CHANGELOG.md** with your changes
4. **Ensure all tests pass**
5. **Submit pull request** with clear description

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
Describe testing performed

## Checklist
- [ ] Code follows project style
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests pass
```

## Project Structure

```
src/
├── data/         # Data loading and preprocessing
├── models/       # Neural network architectures
├── training/     # Training loops and callbacks
├── evaluation/   # Model evaluation and metrics
└── utils/        # Utilities and helpers

scripts/          # Command-line scripts
notebooks/        # Jupyter notebooks
tests/           # Unit tests
```

## Commit Messages

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "Add Morgan fingerprint feature extraction"
git commit -m "Fix bug in learning rate scheduler"
git commit -m "Update README with installation instructions"

# Not as good
git commit -m "Update"
git commit -m "Fix"
git commit -m "Changes"
```

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help create a welcoming environment

## Questions?

If you have questions:
- Check existing issues and discussions
- Open a new issue with the `question` label
- Join our discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
