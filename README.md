# Stock Market Prediction with FinancialBERT

A comprehensive research project implementing and comparing multiple deep learning architectures for stock market prediction using news headlines and FinancialBERT embeddings.

## 📋 Project Overview

This project explores the challenging task of predicting stock price movements based on financial news headlines. We implement a baseline FinancialBERT model and three novel architectures to improve prediction accuracy:

1. **Base Model**: Standard FinancialBERT with simple headline concatenation
2. **Attention-Based Model**: Multi-head attention for dynamic headline importance weighting
3. **Temporal Convolutional Network (TCN)**: Captures temporal dependencies in sequential news
4. **Graph Neural Network (GNN)**: Models relationships between different parts of headlines
5. **Hybrid Multi-Branch Model**: Combines all three novel approaches in parallel

## 🎯 Research Goal

To improve stock market prediction accuracy by leveraging advanced neural architectures that capture different aspects of financial news data:
- **Attention mechanisms** for dynamic importance weighting
- **Temporal modeling** for sequential pattern recognition
- **Graph structures** for relational information processing

## 🏗️ Architecture

### Base Model
- Pre-trained FinancialBERT (ProsusAI/finbert)
- Simple concatenation of top-K relevant headlines
- Linear regression head for percentage change prediction

### Novel Architectures

#### 1. Attention-Based Feature Fusion
```
FinBERT → Multi-Head Self-Attention → Layer Norm → Regression Head
```
- Dynamically weights headline importance
- 8-head attention mechanism
- Residual connections for stable training

#### 2. Temporal Convolutional Network (TCN)
```
FinBERT → TCN Blocks (dilated causal conv) → Global Pooling → Regression
```
- Captures temporal dependencies in news sequences
- Dilated convolutions for larger receptive fields
- Residual connections between blocks

#### 3. Graph Neural Network (GNN)
```
FinBERT → Graph Conv Layers → Hybrid Pooling (mean+max) → Regression
```
- Models token relationships as fully connected graph
- Multiple graph convolutional layers
- Combines mean and max pooling for robust features

#### 4. Hybrid Multi-Branch Model
```
                    ┌─→ Attention Branch (256) ─┐
FinBERT Embeddings ─┼─→ TCN Branch (128) ───────┼─→ Fusion → Regression
                    └─→ GNN Branch (256) ────────┘
```
- Parallel processing through all three branches
- Feature concatenation (640 dims) → Fusion layers (384 → 128)
- Captures complementary patterns from different architectures

## 📊 Dataset

- **Source**: Stock price history paired with financial news headlines
- **Total Samples**: 16,859 (90% train, 10% test)
- **Stocks**: 2,751 companies
- **Features**: Top-K (5-10) most relevant headlines per month
- **Target**: Monthly percentage change in stock price
- **Selection Method**: TF-IDF vectorization + FAISS for relevant headline retrieval

### Data Structure
```
data/
├── stock_data/           # Stock price history (.jsonl)
└── cleaned_training_data.jsonl  # News headlines
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.11+
PyTorch 2.7.0+
CUDA-capable GPU (recommended)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/divyanshu-prakash-rx/Stock_Market_Prediction_NSCAN_with_Event_aware.git
cd Stock_Market_Prediction_NSCAN_with_Event_aware
```

2. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers scikit-learn faiss-cpu numpy pandas matplotlib tqdm
```

3. **Prepare data**
- Extract `data.zip` to create the `data/` directory
- Ensure `data/stock_data/` and `data/cleaned_training_data.jsonl` are present

### Running the Notebook

1. **Activate your Python environment**
```bash
conda activate mltorch311  # or your environment name
```

2. **Open Jupyter Notebook**
```bash
jupyter notebook Improved_stock_market_prediction_FinancialBERT.ipynb
```

3. **Configure training parameters** (Cell 5)
```python
DEVELOPMENT_MODE = False  # Set True for quick testing
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 50
MAX_SEQUENCE_LENGTH = 64
TOP_K_HEADLINES = 5
```

4. **Run cells sequentially** or use "Run All"

## 🧪 Training Configuration

### Recommended Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DEVELOPMENT_MODE` | `False` | Use full dataset |
| `BATCH_SIZE` | `16` | Balance speed/memory |
| `LEARNING_RATE` | `1e-5` to `1e-4` | Higher LR helps with underfitting |
| `NUM_EPOCHS` | `50` | Full training convergence |
| `MAX_SEQUENCE_LENGTH` | `64-128` | Token limit per sample |
| `TOP_K_HEADLINES` | `5-10` | Most relevant headlines |

### Model Training

Each model is trained with:
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Validation**: Test set evaluation every epoch
- **Checkpointing**: Best model saved based on test loss

## 📈 Results & Performance

### Evaluation Metrics
- **MSE (Mean Squared Error)**: Lower is better
- **RMSE (Root Mean Squared Error)**: In percentage change units
- **MAE (Mean Absolute Error)**: Average prediction error
- **R² Score**: Proportion of variance explained (higher is better)

### Expected Performance

Stock price prediction from news alone is extremely challenging. Typical results:

- **R² Score**: 0.08 - 0.30 (8% - 30% variance explained)
- **Correlation**: 0.20 - 0.40 (weak to moderate)
- **Variance Ratio**: 0.30 - 0.50 (predictions vs. actuals)

**Note**: R² of 0.20-0.30 is considered **good** for this task, as markets are influenced by many factors beyond news headlines.

### Diagnostic Analysis

The notebook includes comprehensive diagnostics:
- Prediction variance analysis
- Correlation with actual values
- Underfitting detection
- Recommended actions for improvement

## 🔧 Troubleshooting

### Common Issues

1. **Low R² Score (< 0.1)**
   - Increase learning rate (try 5e-5 or 1e-4)
   - Train for full 50 epochs
   - Unfreeze more BERT layers
   - Increase model capacity

2. **GPU Out of Memory**
   - Reduce `BATCH_SIZE` (try 8 or 4)
   - Reduce `MAX_SEQUENCE_LENGTH`
   - Reduce `TOP_K_HEADLINES`

3. **Training Too Slow**
   - Enable `DEVELOPMENT_MODE` for testing
   - Reduce `NUM_EPOCHS` temporarily
   - Use smaller `TRAIN_SAMPLE_LIMIT`

4. **Model Predicting Constant Values**
   - Check prediction variance ratio
   - Increase learning rate
   - Verify data preprocessing
   - Ensure sufficient training epochs

## 📁 Project Structure

```
Project_NLP/
├── Improved_stock_market_prediction_FinancialBERT.ipynb  # Main notebook
├── README.md                                              # This file
├── data/
│   ├── stock_data/                                        # Price history
│   └── cleaned_training_data.jsonl                        # News headlines
├── model/                                                 # Saved model checkpoints
│   ├── base_model_best.pth
│   ├── attention_model_best.pth
│   ├── tcn_model_best.pth
│   ├── gnn_model_best.pth
│   └── hybrid_model_best.pth
└── data.zip                                               # Compressed data archive
```

## 🔬 Key Features

### 1. Top-K Headline Selection
- TF-IDF vectorization for semantic relevance
- FAISS for efficient similarity search
- Selects most informative headlines per stock/month

### 2. Multiple Model Architectures
- Modular design for easy comparison
- Shared FinBERT encoder for fair evaluation
- Novel fusion strategies for improved performance

### 3. Comprehensive Evaluation
- Multiple metrics (MSE, RMSE, MAE, R²)
- Training/test loss curves
- Predictions vs. actuals visualization
- Stock-specific prediction analysis

### 4. Diagnostic Tools
- Variance ratio analysis
- Correlation measurement
- Underfitting detection
- Actionable recommendations

## 📚 Technical Details

### FinancialBERT
- **Model**: ProsusAI/finbert
- **Parameters**: ~110M total
- **Trainable**: Only regression heads (frozen BERT)
- **Embedding Size**: 768 dimensions

### Tokenization
- WordPiece tokenization
- Special tokens: [CLS], [SEP]
- Padding to max sequence length
- Truncation for long headlines

### Feature Engineering
- Headline concatenation with period separators
- Top-K selection via semantic similarity
- Percentage change normalization

## 🎓 Research Context

This project addresses the fundamental challenge of financial market prediction:

**Why is this difficult?**
- Markets incorporate many factors beyond news
- News sentiment may already be priced in
- Short-term noise dominates signals
- High-frequency trading and algorithmic responses

**What makes this approach novel?**
- Multi-architecture comparison
- Hybrid ensemble approach
- Attention-based feature fusion
- Temporal and graph-based modeling

## 📊 Visualization

The notebook generates:
1. Training/test loss curves for all models
2. Predictions vs. actuals scatter plots
3. Metric comparison bar charts
4. Stock-specific price prediction visualizations (TSLA, AAPL)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional feature engineering
- More sophisticated architectures
- Ensemble methods
- Alternative loss functions
- Multi-task learning approaches

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{stock_prediction_finbert_2024,
  author = {Divyanshu Prakash},
  title = {Stock Market Prediction with FinancialBERT and Novel Architectures},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/divyanshu-prakash-rx/Stock_Market_Prediction_NSCAN_with_Event_aware}
}
```

## 📄 License

This project is available for academic and research purposes.

## 🙏 Acknowledgments

- **FinancialBERT**: ProsusAI for the pre-trained financial domain model
- **Transformers Library**: HuggingFace for the excellent NLP toolkit
- **PyTorch**: For the deep learning framework
- **Research Inspiration**: Based on various financial NLP and time-series forecasting papers

## 📧 Contact

For questions or collaborations:
- GitHub: [@divyanshu-prakash-rx](https://github.com/divyanshu-prakash-rx)
- Repository: [Stock_Market_Prediction_NSCAN_with_Event_aware](https://github.com/divyanshu-prakash-rx/Stock_Market_Prediction_NSCAN_with_Event_aware)

## 🔮 Future Work

- [ ] Add sentiment analysis features
- [ ] Incorporate technical indicators
- [ ] Implement transformer-based time series models
- [ ] Multi-horizon prediction (weekly, monthly)
- [ ] Cross-asset correlation modeling
- [ ] Real-time prediction pipeline
- [ ] Explainability analysis (SHAP, attention visualization)

---

**Note**: Stock market prediction is inherently uncertain. This project is for research and educational purposes only and should not be used for actual trading decisions.
