# CAST-CKT
CAST-CKT: Chaos-Aware Spatio-Temporal and Cross-City Knowledge Transfer for Traffic Flow Prediction: CAST-CKT is an advanced deep learning framework for few-shot traffic forecasting that integrates chaos theory with spatio-temporal graph neural networks. Designed for cross‑city generalization under data scarcity, CAST‑CKT dynamically adapts to varying traffic predictability regimes by extracting chaos‑theoretic features (Lyapunov exponents, fractal dimensions, entropy) and using them to modulate attention, graph construction, and uncertainty estimation.

## Key Features

- **Chaos-Aware Architecture**: Integrates chaos theory principles with deep learning for better modeling of traffic flow dynamics
- **Physics-Informed Features**: Incorporates domain knowledge including degree centrality, neighbor influence, and temporal gradients
- **Multi-Scale Learning**: Captures short-term, medium-term, and long-term temporal dependencies
- **Adaptive Topology Learning**: Dynamically learns spatial relationships in traffic networks
- **Multi-Horizon Prediction**: Supports prediction at multiple time horizons (5min to 2 hours)
- **Domain Adaptation**: Few-shot learning capabilities for cross-city traffic prediction
- **Multiple Datasets**: Support for METR-LA, PEMS-BAY, Shenzhen, and Chengdu traffic datasets


## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

# Clone the repository
git clone https://github.com/yourusername/CAST-CKT.git
cd chaos-traffic-prediction


# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

### Required Dependencies
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
pyyaml>=5.4.0
tqdm>=4.62.0
torch>=2.0.0

## Quick Start

# Train on METR-LA dataset
python main.py --config config.yaml --dataset metr-la

# Train with physics features enabled
python main.py --config config.yaml --dataset metr-la --physics_features

# Domain adaptation training
python main.py --config config.yaml --dataset metr-la --domain_adaptation

### 3. Multi-Dataset Evaluation
# Test on multiple datasets
python main.py --config config.yaml --datasets metr-la,pems-bay,chengdu_m

## Model Architecture

### Core Components

#### 1. **ChaosAnalyzer**
Extracts 20 chaos-theoretic features including:
- Largest Lyapunov Exponent
- Hurst Exponent  
- Sample Entropy
- Correlation Dimension
- Box Counting Dimension
- Recurrence Rate
- Multifractal Spectrum


#### 3. **MultiScaleTemporalEncoder**
Captures temporal dependencies at multiple scales:
- Short-term: Immediate patterns (1-2 steps)
- Medium-term: Local trends (3-6 steps)  
- Long-term: Global patterns (12+ steps)
- Transformer layers for sequence modeling

#### 4. **AdaptiveTopologyLearning**
Dynamically learns spatial graph structure:
- Local and global attention mechanisms
- Distance-based edge weighting
- Chaos-informed adjacency prediction


## Datasets

The framework supports four major traffic datasets:

| Dataset | Nodes | Timesteps | Location | Time Resolution |
|---------|-------|-----------|----------|----------------|
| METR-LA | 207 | 34,272 | Los Angeles | 5 minutes |
| PEMS-BAY | 325 | 52,116 | Bay Area | 5 minutes |
| Chengdu-M | 524 | 17,280 | Chengdu | 15 minutes |
| Shenzhen | 627 | 17,280 | Shenzhen | 15 minutes |

### Dataset Structure
```
data/
├── metr-la/
│   ├── dataset.npy           # Traffic flow data [34272, 207, 1]
│   ├── matrix.npy           # Adjacency matrix [207, 207]
│   ├── node_features.npy    # Optional: Node attributes [207, F]
│   └── metadata.json       # Optional: Dataset metadata
├── pems-bay/
│   ├── dataset.npy         # Traffic flow data [52116, 325, 1]
│   ├── matrix.npy         # Adjacency matrix [325, 325]
│   ├── node_features.npy  # Optional: Node attributes [325, F]
│   └── metadata.json     # Optional: Dataset metadata
├── chengdu_m/
│   ├── dataset.npy       # Traffic flow data [17280, 524, 1]
│   ├── matrix.npy       # Adjacency matrix [524, 524]
│   ├── node_features.npy # Optional: Node attributes [524, F]
│   └── metadata.json   # Optional: Dataset metadata
├── shenzhen/
│   ├── dataset.npy     # Traffic flow data [17280, 627, 1]
│   ├── matrix.npy     # Adjacency matrix [627, 627]
│   ├── node_features.npy # Optional: Node attributes [627, F]
│   └── metadata.json # Optional: Dataset metadata
└── cache/             # Auto-generated cache directory
    ├── enhanced_*.pkl # Preprocessed datasets with physics features
    └── features_*.pkl # Cached chaos features
```

## Configuration

The system uses YAML configuration files. Key sections:

### Model Configuration
```yaml
model:
  hidden_dim: 16        # Hidden layer dimension
  chaos_dim: 20         # Chaos feature dimension
  num_heads: 8          # Attention heads
  dropout: 0.1          # Dropout rate
  noise_std: 0.005      # Input noise for regularization

### Training Configuration
```yaml
training:
  epochs: 500           # Training epochs
  batch_size: 32        # Batch size
  learning_rate: 0.0005 # Learning rate
  add_physics_features: true  # Enable physics features
```

### Domain Adaptation
```yaml
few_shot:
  enabled: true         # Enable domain adaptation
  adaptation_steps: 5   # Inner loop steps
  support_size: 8       # Support set size
  query_size: 12        # Query set size
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{chaos_traffic_2026,
  title={Chaos-Aware Deep Learning for Traffic Flow Prediction: Integrating Physics-Informed Features and Adaptive Topology Learning},
  author={Fofanah et al.},
  journal={arXiv preprint arXiv:2026.xxxxx},
  year={2026}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Traffic datasets provided by CalTrans (METR-LA, PEMS-BAY) and respective transportation authorities
- Chaos theory implementations inspired by established nonlinear dynamics literature
- PyTorch Geometric community for graph neural network foundations

## Contact

- **Email**: a.fofanah@griffith.edu.au or dmitripeter.fofanah@gmail.com


