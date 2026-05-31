# Web Application Guide

## Running the BBB Penetration Predictor Web App

### Local Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. **Access the app:**
   Open your browser and go to `http://localhost:8501`

### Codespaces Setup

1. **In GitHub Codespaces terminal:**
   ```bash
   pip install -r requirements.txt
   streamlit run app.py --server.port 8501
   ```

2. **Access via forwarded port:**
   - Click on the "Ports" tab in Codespaces
   - Find port 8501 and click "Open in Browser"

### Deployment Options

#### Option 1: Streamlit Cloud (Recommended)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy! Free hosting for public repos

#### Option 2: Hugging Face Spaces
1. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select "Streamlit" as the SDK
3. Upload your code
4. Add `models/checkpoints/best_model.pth` and `data/processed/scaler.pkl`

#### Option 3: Railway/Render
- Both support Streamlit deployments
- Follow their respective documentation

## Features

✨ **Interactive molecule input** - Enter SMILES strings or select examples  
🔬 **Real-time predictions** - Instant BBB penetration predictions  
📊 **Confidence scores** - See model confidence and probability breakdown  
🧬 **Molecular visualization** - View 2D structure of input molecules  
📈 **Molecular properties** - Calculate key physicochemical descriptors  

## Usage Tips

### Example SMILES Strings

- **Aspirin:** `CC(=O)OC1=CC=CC=C1C(=O)O`
- **Caffeine:** `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
- **Dopamine:** `C1=CC(=C(C=C1CCN)O)O`
- **Ibuprofen:** `CC(C)CC1=CC=C(C=C1)C(C)C(=O)O`
- **Nicotine:** `CN1CCCC1C2=CN=CC=C2`

### For Hackathon Demo

1. **Start with the app running**
2. **Show the About section** - explain the problem and model performance
3. **Demo with examples** - use the sidebar examples
4. **Explain confidence scores** - highlight 93.2% precision
5. **Show molecular properties** - demonstrate chemistry integration
6. **Test custom molecules** - show it works on any SMILES

## Troubleshooting

**Issue: RDKit not found**
```bash
pip install rdkit
# or
conda install -c conda-forge rdkit
```

**Issue: Model not found**
- Ensure `models/checkpoints/best_model.pth` exists
- Ensure `data/processed/scaler.pkl` exists
- Check paths in `config/config.yaml`

**Issue: Port already in use**
```bash
streamlit run app.py --server.port 8502
```
