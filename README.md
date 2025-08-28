# Geometric Multi-Resolution Analysis (GMRA) in Python

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4.1](https://img.shields.io/badge/pytorch-2.4.1+cu121-green.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A comprehensive Python implementation of Geometric Multi-Resolution Analysis (GMRA) for high-dimensional data analysis, dimensionality reduction, and multi-scale geometric feature extraction.**

## 🎯 Project Overview

This repository implements GMRA, a mathematical framework for analyzing high-dimensional data by constructing multi-scale geometric structures. GMRA is particularly powerful for:

- **Dimensionality Reduction**: Finding low-dimensional geometric structures in high-dimensional data
- **Multi-Scale Analysis**: Analyzing data at multiple resolutions simultaneously
- **Manifold Learning**: Discovering underlying geometric structures in complex datasets
- **Feature Extraction**: Generating hierarchical, interpretable features for machine learning

## 🏗️ Architecture

The GMRA framework consists of several key components:

### Core Components (`src/`)

- **`covertree.py`** (39KB): Cover tree implementation for efficient nearest neighbor search
- **`dyadictree.py`** (37KB): Dyadic tree structure for multi-resolution analysis
- **`dyadictreenode.py`** (3.7KB): Individual nodes in the dyadic tree
- **`wavelettree.py`** (15KB): Wavelet tree implementation for signal processing
- **`helpers.py`** (5.6KB): Utility functions for tree operations
- **`utils.py`** (7.6KB): General utility functions and data processing

### Experimental Framework (`experiments/`)

- **`classifications/`**: Classification experiments using GMRA features
- **`mnist/`**: MNIST dataset experiments and analysis
- **`cifar10/`**: CIFAR-10 dataset experiments
- **`medical_img/`**: Medical image analysis experiments

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.8
- **PyTorch**: 2.4.1+cu121 (CUDA support recommended)
- **CUDA**: 12.1 (if using GPU acceleration)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/klimantas/gmra_archive.git
   cd gmra_archive
   git checkout remove-wavelettree
   ```

2. **Create conda environment:**
   ```bash
   conda create -n gmra-env python=3.8
   conda activate gmra-env
   ```

3. **Install PyTorch:**
   ```bash
   pip install torch==2.4.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install additional dependencies:**
   ```bash
   pip install numpy scipy scikit-learn matplotlib seaborn pandas tqdm pyyaml hydra-core tensorboard
   ```

### Basic Usage

```python
import numpy as np
from src.covertree import CoverTree
from src.dyadictree import DyadicTree

# Create sample high-dimensional data
X = np.random.randn(1000, 100)  # 1000 points in 100D space

# Build cover tree
cover_tree = CoverTree(X)

# Build dyadic tree with GMRA
gmra_tree = DyadicTree(
    cover_tree, 
    X=X,
    manifold_dims=10,  # Target manifold dimension
    max_dim=20,        # Maximum dimension for analysis
    thresholds=0.5,    # Threshold for tree construction
    precisions=1e-2    # Precision for geometric analysis
)

# Extract GMRA features
features = gmra_tree.extract_features(X)
print(f"Original dimension: {X.shape[1]}")
print(f"GMRA features dimension: {features.shape[1]}")
```

## 📊 Experiments

### Classification Experiments

The `experiments/classifications/` directory contains comprehensive classification experiments:

```bash
cd experiments/classifications

# Run basic MNIST experiment
python gmra_classification_experiment.py

# Run with custom configuration
python gmra_classification_experiment.py --config-name=config_small

# Run CIFAR-10 experiment
python gmra_classification_experiment.py --config-name=config_cifar10

# Hyperparameter tuning
python gmra_classification_experiment.py --config-name=config_tuning
```

**Key Features:**
- **Multiple Classifiers**: Random Forest, SVM, KNN, Naive Bayes, Neural Networks
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, timing
- **Hydra Configuration**: Easy experiment management and hyperparameter tuning
- **TensorBoard Logging**: Real-time visualization of results
- **Multiple Datasets**: MNIST, CIFAR-10, and custom datasets

### MNIST Experiments

The `experiments/mnist/` directory contains Jupyter notebooks for detailed analysis:

- **`mnist.ipynb`**: Comprehensive GMRA analysis of MNIST
- **`mnist_batch.ipynb`**: Batch processing experiments
- **`mnist_pca.ipynb`**: PCA comparison experiments

### CIFAR-10 Experiments

Advanced experiments with the CIFAR-10 dataset for image classification:

```bash
# Quick CIFAR-10 test
python gmra_classification_experiment.py --config-name=config_cifar10_small

# Full CIFAR-10 experiment
python gmra_classification_experiment.py --config-name=config_cifar10
```

## 🔬 DARPA CASTLE Project Integration

This repository is being used for the **DARPA CASTLE project** in collaboration with Kevin Limanta. The project focuses on:

### Research Objectives
1. **Graph Neural Network Embeddings**: Use GCN/GraphSAGE to generate embeddings in Euclidean space
2. **GMRA Dimensionality Reduction**: Apply GMRA to find lower-dimensional geometric structures
3. **Classification Comparison**: Compare performance of GMRA embeddings vs. original embeddings
4. **Multi-Scale Analysis**: Leverage GMRA's multi-resolution capabilities for graph analysis

### Workflow for CASTLE
```python
# 1. Generate graph embeddings using GCN/GraphSAGE
graph_embeddings = generate_graph_embeddings(graph_data)

# 2. Apply GMRA for dimensionality reduction
gmra_features = apply_gmra(graph_embeddings, manifold_dims=64)

# 3. Perform classification tasks
original_accuracy = classify(graph_embeddings)
gmra_accuracy = classify(gmra_features)

# 4. Compare results
print(f"Original accuracy: {original_accuracy:.3f}")
print(f"GMRA accuracy: {gmra_accuracy:.3f}")
```

## 📁 Project Structure

```
gmra/
├── src/                          # Core GMRA implementation
│   ├── covertree.py             # Cover tree for nearest neighbor search
│   ├── dyadictree.py            # Main GMRA dyadic tree
│   ├── dyadictreenode.py        # Tree node implementation
│   ├── wavelettree.py           # Wavelet tree processing
│   ├── helpers.py               # Tree operation utilities
│   └── utils.py                 # General utilities
├── experiments/                  # Experimental framework
│   ├── classifications/         # Classification experiments
│   │   ├── gmra_classification_experiment.py  # Main experiment script
│   │   ├── config.yaml          # Default configuration
│   │   ├── config_cifar10.yaml  # CIFAR-10 specific config
│   │   └── requirements.txt     # Dependencies
│   ├── mnist/                   # MNIST dataset experiments
│   ├── cifar10/                 # CIFAR-10 experiments
│   └── medical_img/             # Medical image analysis
├── datasets/                     # Dataset storage (symlink recommended)
├── tests/                        # Unit tests
└── README.md                     # This file
```

## ⚙️ Configuration

### GMRA Parameters

- **`manifold_dims`**: Target dimension for the underlying manifold
- **`max_dim`**: Maximum dimension for geometric analysis
- **`thresholds`**: Threshold values for tree construction
- **`precisions`**: Precision values for geometric calculations

### Experiment Configuration

Experiments use Hydra for configuration management. Key parameters include:

```yaml
data:
  num_points: 2000          # Number of data points to use
  dataset: "mnist"          # Dataset name

gmra:
  leafsize: 10              # Leaf size for tree construction
  tree_subset_ratio: 0.1    # Ratio of data for tree building
  min_points: 50            # Minimum points per node

classifiers:
  random_forest:
    n_estimators: 100       # Number of trees
  svm:
    C: 1.0                  # Regularization parameter
```

## 📈 Performance and Scaling

### Computational Complexity
- **Cover Tree Construction**: O(n log n) average case
- **GMRA Tree Building**: O(n log n) for balanced trees
- **Feature Extraction**: O(n) for single pass through tree

### Memory Requirements
- **Cover Tree**: O(n) storage for tree structure
- **GMRA Features**: O(n × manifold_dims) for extracted features
- **Total Memory**: Scales linearly with dataset size

### GPU Acceleration
- **PyTorch Integration**: Leverages CUDA for matrix operations
- **Batch Processing**: Efficient batch processing of large datasets
- **Memory Management**: Automatic GPU memory optimization

## 🔍 Advanced Usage

### Custom Datasets

```python
from src.dyadictree import DyadicTree
from src.covertree import CoverTree

# Load your custom dataset
X = load_custom_data()

# Build GMRA tree
cover_tree = CoverTree(X)
gmra_tree = DyadicTree(cover_tree, X=X, manifold_dims=32)

# Extract features at different scales
features_level_1 = gmra_tree.extract_features_at_level(X, level=1)
features_level_2 = gmra_tree.extract_features_at_level(X, level=2)
```

### Multi-Scale Analysis

```python
# Analyze data at multiple resolutions
scales = [1, 2, 3, 4]
multi_scale_features = {}

for scale in scales:
    features = gmra_tree.extract_features_at_level(X, level=scale)
    multi_scale_features[f'level_{scale}'] = features

# Combine multi-scale features
combined_features = np.hstack(list(multi_scale_features.values()))
```

### Custom Distance Functions

```python
# Implement custom distance metrics
def custom_distance(x, y):
    return np.linalg.norm(x - y, ord=1)  # L1 distance

# Use in cover tree construction
cover_tree = CoverTree(X, distance_func=custom_distance)
```

## 🧪 Testing

Run the test suite to ensure everything is working correctly:

```bash
cd tests
python -m pytest test_*.py -v
```

## 📚 References

### Core Papers
- **GMRA Foundation**: [Multiscale Geometric Methods for Data Sets II: Geometric Multi-Resolution Analysis](https://arxiv.org/abs/1105.4924)
- **Cover Trees**: [Cover Trees for Nearest Neighbor](https://dl.acm.org/doi/10.1145/1143844.1143859)

### Related Work
- **Manifold Learning**: Isomap, LLE, t-SNE
- **Multi-Scale Analysis**: Wavelets, Laplacian Eigenmaps
- **Dimensionality Reduction**: PCA, UMAP, PHATE

## 🤝 Contributing

This project is part of ongoing research. For contributions:

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests if applicable**
5. **Submit a pull request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Kevin Limanta**: Original repository and GMRA implementation
- **DARPA CASTLE**: Research project funding and direction
- **Research Community**: Contributions to geometric analysis and manifold learning

## 📞 Contact

- **Repository**: [https://github.com/klimantas/gmra_archive](https://github.com/klimantas/gmra_archive)
- **DARPA CASTLE Project**: Collaboration with Kevin Limanta
- **Research Area**: Geometric Multi-Resolution Analysis for High-Dimensional Data

---

**Note**: This repository is actively maintained for research purposes. For questions about the DARPA CASTLE project or GMRA implementation, please contact the research team.