# GMRA Classification Experiment with Hydra

This experiment evaluates the effectiveness of GMRA (Geometric Multi-Resolution Analysis) feature extraction for classification tasks using MNIST data. It uses Hydra for configuration management and hyperparameter tuning.

## Features

- **Hydra Configuration**: Easy experiment configuration and hyperparameter management
- **GMRA Feature Extraction**: Uses `fgwt_batch` to extract GMRA coefficients
- **Baseline Comparisons**: Compares against raw pixels and PCA features
- **Multiple Classifiers**: Tests Random Forest, SVM, KNN, Naive Bayes, and Neural Network
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, and timing
- **TensorBoard Logging**: Real-time metrics visualization
- **Text File Logging**: Detailed results saved to files
- **Visualization**: Heatmaps and bar charts for results comparison
- **Configuration Tracking**: Automatic saving of experiment configurations

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure the GMRA source code is available:
   ```bash
   # The script expects the following structure:
   # ../../src/covertree.py
   # ../../src/dyadictree.py
   # ../../src/dyadictreenode.py
   # ../../src/utils.py
   ```

## Usage

### Basic Usage
```bash
python gmra_classification_experiment.py
```

### Using Different Configurations
```bash
# Use small configuration for quick testing
python gmra_classification_experiment.py --config-name=config_small

# Use tuning configuration for hyperparameter exploration
python gmra_classification_experiment.py --config-name=config_tuning

# Use CIFAR-10 dataset
python gmra_classification_experiment.py --config-name=config_cifar10

# Use small CIFAR-10 configuration for quick testing
python gmra_classification_experiment.py --config-name=config_cifar10_small
```

### Override Configuration Parameters
```bash
# Override specific parameters
python gmra_classification_experiment.py data.num_points=2000 gmra.leafsize=5

# Override GMRA tree building parameters
python gmra_classification_experiment.py gmra.tree_subset_ratio=0.05 gmra.min_points=20

# Override classifier parameters
python gmra_classification_experiment.py classifiers.random_forest.n_estimators=200

# Change experiment name and seed
python gmra_classification_experiment.py experiment.name="custom_experiment" experiment.seed=123
```

### Multiple Experiments (Hydra Multirun)
```bash
# Run multiple experiments with different seeds
python gmra_classification_experiment.py --multirun experiment.seed=42,123,456

# Run with different leafsize values
python gmra_classification_experiment.py --multirun gmra.leafsize=5,10,20

# Run with different datasets sizes
python gmra_classification_experiment.py --multirun data.num_points=500,1000,2000

# Compare MNIST vs CIFAR-10
python gmra_classification_experiment.py --multirun --config-name=config_small,config_cifar10_small

# CIFAR-10 hyperparameter sweep
python gmra_classification_experiment.py --config-name=config_cifar10 --multirun gmra.tree_subset_ratio=0.05,0.1,0.2
```

## Configuration Files

### Main Configuration (`config.yaml`)
The main configuration file contains all default parameters:

```yaml
experiment:
  name: "gmra_classification"
  description: "GMRA feature extraction for MNIST classification"
  seed: 42

data:
  dataset: "mnist"
  num_points: 1000
  flatten: true
  test_size: 0.2

gmra:
  leafsize: 10
  manifold_dims: 0
  max_dim: null
  thresholds: 0.0
  precisions: 1e-2
  min_points: 10
  tree_subset_ratio: 0.1  # Use 10% of training data for tree building

classifiers:
  random_forest:
    n_estimators: 100
    max_depth: null
  # ... other classifiers
```

### Specialized Configurations

- **`config_small.yaml`**: Quick testing with smaller dataset and fewer classifiers
- **`config_tuning.yaml`**: Hyperparameter tuning with different parameter values
- **`config_cifar10.yaml`**: CIFAR-10 dataset comparison (5000 samples)
- **`config_cifar10_small.yaml`**: Quick CIFAR-10 testing (1000 samples)

### Creating Custom Configurations

Create your own configuration file:

```yaml
# config_custom.yaml
defaults:
  - config

experiment:
  name: "my_custom_experiment"
  
data:
  num_points: 1500
  
gmra:
  leafsize: 15
  min_points: 15
  tree_subset_ratio: 0.2  # Use 20% of training data for tree building
```

Then run:
```bash
python gmra_classification_experiment.py --config-name=config_custom
```

## Tree Building with Subset Data

For memory efficiency and faster tree construction, the experiment uses only a subset of the training data to build the GMRA tree. The `tree_subset_ratio` parameter controls what fraction of the training data is used for tree construction:

- **Default**: 10% of training data (`tree_subset_ratio: 0.1`)
- **Small config**: 20% of training data (fewer samples overall)
- **Tuning config**: 5% of training data (more samples available)

The tree is built using this subset, but feature extraction is performed on the full training and test sets. This approach:
- Reduces memory usage during tree construction
- Speeds up tree building process
- Still captures the underlying data structure for feature extraction
- Allows processing of larger datasets

You can adjust this parameter based on your memory constraints and performance needs:
```bash
# Use 5% of training data for tree building
python gmra_classification_experiment.py gmra.tree_subset_ratio=0.05

# Use 20% of training data for tree building
python gmra_classification_experiment.py gmra.tree_subset_ratio=0.20
```

## Output Files

The experiment creates timestamped output files in the configured log directory:

1. **Log Files**: `gmra_classification_YYYYMMDD_HHMMSS.log`
   - Detailed execution logs with Hydra configuration
   - Progress tracking and hyperparameter values
   - Error messages and debugging information

2. **Configuration Files**: `config_YYYYMMDD_HHMMSS.yaml`
   - Complete configuration used for the experiment
   - Enables exact reproduction of results

3. **Results Files**: `results_YYYYMMDD_HHMMSS.txt`
   - Comprehensive results summary with configuration details
   - Performance metrics for all feature types and classifiers
   - Best performing combinations

4. **Plot Files**: `results_plot_YYYYMMDD_HHMMSS.png` (if `logging.save_plots: true`)
   - High-resolution result visualizations
   - Saved automatically when configured

5. **TensorBoard Logs**: `logs/tensorboard/experiment_name_YYYYMMDD_HHMMSS/`
   - Real-time metrics visualization
   - Hyperparameter tracking

## TensorBoard Visualization

To view TensorBoard logs:
```bash
tensorboard --logdir=logs/tensorboard
```

Then open http://localhost:6006 in your browser.

## Metrics Tracked

### Tree Construction
- Number of nodes
- Number of leaf nodes
- Tree height
- Leaf size parameters
- Tree subset ratio and sample count (for memory efficiency)

### Feature Extraction
- GMRA feature dimensionality
- PCA feature dimensionality
- PCA explained variance
- Extraction timing

### Classification Performance
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Training time
- Prediction time

## Hydra Features

### Experiment Tracking
Hydra automatically creates separate output directories for each experiment run:
```
outputs/
├── 2024-01-15/
│   ├── 10-30-45/          # Timestamp directory
│   │   ├── .hydra/        # Hydra metadata
│   │   ├── config.yaml    # Resolved configuration
│   │   └── logs/          # Experiment outputs
│   └── 11-45-20/
└── 2024-01-16/
```

### Multirun Experiments
Run multiple experiments with different parameters:
```bash
# Test different random seeds
python gmra_classification_experiment.py --multirun experiment.seed=42,123,456,789

# Test different GMRA parameters
python gmra_classification_experiment.py --multirun gmra.leafsize=5,10,20 gmra.min_points=5,10,20

# Test different dataset sizes
python gmra_classification_experiment.py --multirun data.num_points=500,1000,2000
```

### Configuration Composition
Combine different configuration aspects:
```bash
# Use small dataset with tuning parameters
python gmra_classification_experiment.py --config-name=config_small gmra.leafsize=5

# Override multiple classifier parameters
python gmra_classification_experiment.py \
  classifiers.random_forest.n_estimators=200 \
  classifiers.svm.C=10.0 \
  classifiers.neural_network.hidden_layer_sizes=[200,100]
```

## TensorBoard Visualization

To view TensorBoard logs:
```bash
# View logs for specific experiment
tensorboard --logdir=logs/tensorboard/gmra_classification_20240115_103045

# View all experiments
tensorboard --logdir=logs/tensorboard

# Or if using Hydra outputs directory
tensorboard --logdir=outputs/2024-01-15/10-30-45/logs/tensorboard
```

## Best Practices

### 1. Experiment Organization
- Use descriptive experiment names: `experiment.name="gmra_leafsize_comparison"`
- Include relevant parameters in the name for easy identification
- Use consistent random seeds for reproducibility

### 2. Hyperparameter Tuning
- Start with `config_small.yaml` for quick parameter exploration
- Use multirun for systematic parameter sweeps
- Save successful configurations for future reference

### 3. Result Analysis
- Compare TensorBoard metrics across different runs
- Use the generated plots for publication-ready figures
- Review text result files for detailed numerical comparisons

### 4. Debugging
- Set `logging.level: DEBUG` for detailed execution logs
- Use smaller datasets (`data.num_points=100`) for quick debugging
- Enable plot saving for offline analysis

## Example Workflows

### Quick Testing
```bash
# Fast experiment for code testing
python gmra_classification_experiment.py --config-name=config_small data.num_points=100
```

### Hyperparameter Search
```bash
# Systematic parameter exploration
python gmra_classification_experiment.py --multirun \
  gmra.leafsize=5,10,20 \
  gmra.min_points=5,10,20 \
  experiment.seed=42,123,456
```

### Production Run
```bash
# Full experiment with optimal parameters
python gmra_classification_experiment.py \
  data.num_points=5000 \
  gmra.leafsize=5 \
  gmra.min_points=5 \
  classifiers.random_forest.n_estimators=200 \
  logging.save_plots=true
```

## Troubleshooting

1. **Hydra configuration errors**: Check YAML syntax and parameter names
2. **TensorBoard not available**: Install with `pip install tensorboard torch`
3. **Memory issues**: Reduce `data.num_points` or increase `gmra.leafsize`
4. **Import errors**: Ensure GMRA source files are in `../../src/`
5. **Multirun failures**: Check that all parameter combinations are valid

## Configuration Reference

See `config.yaml` for all available parameters and their descriptions. Use `--help` to see Hydra-specific options:

```bash
python gmra_classification_experiment.py --help
```
