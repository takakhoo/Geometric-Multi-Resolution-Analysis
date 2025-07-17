#!/usr/bin/env python3
"""
GMRA Classification Experiment with Hydra Configuration

This script evaluates the effectiveness of GMRA fgwt_batch coefficients for classification tasks.
It compares GMRA feature extraction against raw pixel features and other dimensionality reduction methods.
Uses Hydra for configuration management and hyperparameter tuning.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
import time
import os
from datetime import datetime
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

# Hydra for configuration management
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra

# TensorBoard logging
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("TensorBoard not available. Install with: pip install tensorboard torch")
    TENSORBOARD_AVAILABLE = False

# Configure logging with file handler
def setup_logging(cfg: DictConfig):
    """Setup logging to both console and file with Hydra configuration."""
    log_dir = cfg.logging.log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for unique log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"gmra_classification_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to file: {log_file}")
    logger.info(f"Experiment: {cfg.experiment.name}")
    logger.info(f"Description: {cfg.experiment.description}")
    
    return logger, log_file

# Set random seeds for reproducibility
np.random.seed(42)

# Add project root to path
sys.path.insert(0, '../..')
from src.covertree import CoverTree 
from src.dyadictree import DyadicTree
from src.dyadictreenode import DyadicTreeNode
from src.utils import *

def load_mnist_with_labels(num_points=1000, flatten=True):
    """Load MNIST data with labels for classification."""
    from torchvision import datasets, transforms
    
    # Get a logger for this function
    logger = logging.getLogger(__name__)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load training data
    mnist_train = datasets.MNIST(root='../../datasets', train=True, download=True, transform=transform)
    
    # Extract images and labels
    X = []
    y = []
    
    for i, (img, label) in enumerate(mnist_train):
        # If num_points is None, use all data points
        if num_points is not None and i >= num_points:
            break
        X.append(np.array(img.numpy()))
        y.append(label)
    
    X = np.stack(X)
    y = np.array(y)
    
    original_shape = X.shape
    if flatten:
        X = X.reshape(X.shape[0], -1)
    
    logger.info(f"Loaded {len(X)} MNIST samples with shape {X.shape}")
    logger.info(f"Label distribution: {np.bincount(y)}")
    
    return X, y, original_shape

def load_cifar10_with_labels(num_points=1000, flatten=True):
    """Load CIFAR-10 data with labels for classification."""
    from torchvision import datasets, transforms
    
    # Get a logger for this function
    logger = logging.getLogger(__name__)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load training data
    cifar10_train = datasets.CIFAR10(root='../../datasets', train=True, download=True, transform=transform)
    
    # Extract images and labels
    X = []
    y = []
    
    for i, (img, label) in enumerate(cifar10_train):
        # If num_points is None, use all data points
        if num_points is not None and i >= num_points:
            break
        X.append(np.array(img.numpy()))
        y.append(label)
    
    X = np.stack(X)
    y = np.array(y)
    
    original_shape = X.shape
    if flatten:
        X = X.reshape(X.shape[0], -1)
    
    logger.info(f"Loaded {len(X)} CIFAR-10 samples with shape {X.shape}")
    logger.info(f"Label distribution: {np.bincount(y)}")
    
    # CIFAR-10 class names for reference
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    logger.info(f"CIFAR-10 classes: {class_names}")
    
    return X, y, original_shape

class GMRAClassificationExperiment:
    """Class to run GMRA classification experiments with Hydra configuration."""
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize the experiment with Hydra configuration.
        
        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        
        # Set random seed
        np.random.seed(cfg.experiment.seed)
        
        # Setup logging
        self.logger, self.log_file = setup_logging(cfg)
        
        # Setup TensorBoard logging
        if cfg.tensorboard.enabled and TENSORBOARD_AVAILABLE:
            tb_log_dir = os.path.join(cfg.tensorboard.log_dir, 
                                     f"{cfg.experiment.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.tb_writer = SummaryWriter(tb_log_dir)
            self.logger.info(f"TensorBoard logging to: {tb_log_dir}")
        else:
            self.tb_writer = None
            if cfg.tensorboard.enabled:
                self.logger.warning("TensorBoard not available - metrics will not be logged to TensorBoard")
        
        # Load data based on dataset configuration
        if cfg.data.dataset.lower() == "mnist":
            self.X, self.y, self.original_shape = load_mnist_with_labels(
                num_points=cfg.data.num_points, 
                flatten=cfg.data.flatten
            )
        elif cfg.data.dataset.lower() == "cifar10":
            self.X, self.y, self.original_shape = load_cifar10_with_labels(
                num_points=cfg.data.num_points, 
                flatten=cfg.data.flatten
            )
        else:
            raise ValueError(f"Unsupported dataset: {cfg.data.dataset}. Supported datasets: mnist, cifar10")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=cfg.data.test_size, 
            random_state=cfg.experiment.seed, 
            stratify=self.y
        )
        
        self.logger.info(f"Training set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        
        # Initialize feature extractors
        self.gmra_tree = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=cfg.pca.n_components)
        
        # Results storage
        self.results = {}
        self.experiment_step = 0
        
        # Save configuration
        self.save_config()
        
    def save_config(self):
        """Save the current configuration to file."""
        config_file = os.path.join(self.cfg.logging.log_dir, 
                                  f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        with open(config_file, 'w') as f:
            OmegaConf.save(self.cfg, f)
        
        self.logger.info(f"Configuration saved to: {config_file}")
        
    def build_gmra_tree(self):
        """Build and fit GMRA tree on training data using config parameters."""
        self.logger.info("Building GMRA tree...")
        
        cfg = self.cfg.gmra
        max_dim = cfg.max_dim if cfg.max_dim is not None else self.X_train.shape[-1]
        
        # Use subset of training data for tree building
        tree_subset_ratio = cfg.tree_subset_ratio
        n_tree_samples = int(len(self.X_train) * tree_subset_ratio)
        
        # Create random indices for tree building subset
        np.random.seed(self.cfg.experiment.seed)
        tree_indices = np.random.choice(len(self.X_train), n_tree_samples, replace=False)
        X_tree = self.X_train[tree_indices]
        
        self.logger.info(f"Using {n_tree_samples} samples ({tree_subset_ratio*100:.1f}%) for tree building from {len(self.X_train)} training samples")
        
        # Create cover tree
        cover_tree = CoverTree(X_tree, euclidean, leafsize=cfg.leafsize)
        
        # Create dyadic tree
        self.gmra_tree = DyadicTree(
            cover_tree=cover_tree,
            manifold_dims=cfg.manifold_dims,
            max_dim=max_dim,
            thresholds=cfg.thresholds,
            precisions=cfg.precisions,
            inverse=True
        )
        
        # Prune tree
        self.logger.info(f"Pruning nodes with fewer than {cfg.min_points} points...")
        self.gmra_tree.prune_tree_min_point(cfg.min_points)
        
        # Fit the tree
        self.logger.info("Fitting GMRA tree...")
        self.gmra_tree.fit(X_tree)
        
        # Log tree statistics
        all_nodes = self.gmra_tree.get_all_nodes()
        all_leafs = self.gmra_tree.get_all_leafs()
        self.logger.info(f"Tree built: {len(all_nodes)} nodes, {len(all_leafs)} leafs, height: {self.gmra_tree.height}")
        
        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar('Tree/num_nodes', len(all_nodes), self.experiment_step)
            self.tb_writer.add_scalar('Tree/num_leafs', len(all_leafs), self.experiment_step)
            self.tb_writer.add_scalar('Tree/height', self.gmra_tree.height, self.experiment_step)
            self.tb_writer.add_scalar('Tree/tree_subset_samples', n_tree_samples, self.experiment_step)
            self.tb_writer.add_scalar('Tree/tree_subset_ratio', tree_subset_ratio, self.experiment_step)
            self.tb_writer.add_scalar('Tree/leafsize', cfg.leafsize, self.experiment_step)
            self.tb_writer.add_scalar('Tree/min_points', cfg.min_points, self.experiment_step)
        
    def extract_gmra_features(self, X):
        """Extract GMRA features using fgwt_batch."""
        if self.gmra_tree is None:
            raise ValueError("GMRA tree not built. Call build_gmra_tree() first.")
        
        self.logger.info("Extracting GMRA features...")
        start_time = time.time()
        
        # Extract features using batch processing
        gmra_features = self.gmra_tree.fgwt_batch(X)
        
        extraction_time = time.time() - start_time
        self.logger.info(f"GMRA feature extraction completed in {extraction_time:.2f}s")
        self.logger.info(f"GMRA features shape: {gmra_features.shape}")
        
        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar('Features/GMRA_extraction_time', extraction_time, self.experiment_step)
            self.tb_writer.add_scalar('Features/GMRA_feature_dim', gmra_features.shape[1], self.experiment_step)
        
        return gmra_features
    
    def extract_pca_features(self, X_train, X_test):
        """Extract PCA features."""
        self.logger.info("Extracting PCA features...")
        
        # Fit PCA on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        explained_variance = self.pca.explained_variance_ratio_.sum()
        self.logger.info(f"PCA features shape: {X_train_pca.shape}, explained variance: {explained_variance:.3f}")
        
        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar('Features/PCA_feature_dim', X_train_pca.shape[1], self.experiment_step)
            self.tb_writer.add_scalar('Features/PCA_explained_variance', explained_variance, self.experiment_step)
        
        return X_train_pca, X_test_pca
    
    def train_and_evaluate_classifier(self, X_train, X_test, y_train, y_test, 
                                    classifier_name, classifier):
        """Train and evaluate a classifier."""
        self.logger.info(f"Training {classifier_name}...")
        
        start_time = time.time()
        classifier.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        y_pred = classifier.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        self.logger.info(f"{classifier_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar(f'Metrics/{classifier_name}_accuracy', accuracy, self.experiment_step)
            self.tb_writer.add_scalar(f'Metrics/{classifier_name}_f1', f1, self.experiment_step)
            self.tb_writer.add_scalar(f'Metrics/{classifier_name}_precision', precision, self.experiment_step)
            self.tb_writer.add_scalar(f'Metrics/{classifier_name}_recall', recall, self.experiment_step)
            self.tb_writer.add_scalar(f'Timing/{classifier_name}_training_time', training_time, self.experiment_step)
            self.tb_writer.add_scalar(f'Timing/{classifier_name}_prediction_time', prediction_time, self.experiment_step)
        
        return results
    
    def run_classification_experiment(self):
        """Run the complete classification experiment using Hydra configuration."""
        self.logger.info("="*60)
        self.logger.info("GMRA CLASSIFICATION EXPERIMENT")
        self.logger.info("="*60)
        
        # Build GMRA tree
        self.build_gmra_tree()
        
        # Extract features
        self.logger.info("\nExtracting features...")
        
        # 1. Raw pixel features
        X_train_raw = self.X_train
        X_test_raw = self.X_test
        
        # 2. GMRA features
        X_train_gmra = self.extract_gmra_features(self.X_train)
        X_test_gmra = self.extract_gmra_features(self.X_test)
        
        # 3. PCA features
        X_train_pca, X_test_pca = self.extract_pca_features(self.X_train, self.X_test)
        
        # Define classifiers from config
        classifiers = self.create_classifiers()
        
        # Feature sets
        feature_sets = {
            'Raw Pixels': (X_train_raw, X_test_raw),
            'GMRA': (X_train_gmra, X_test_gmra),
            'PCA': (X_train_pca, X_test_pca)
        }
        
        # Run experiments
        self.logger.info("\nRunning classification experiments...")
        results = {}
        
        for feature_name, (X_tr, X_te) in feature_sets.items():
            results[feature_name] = {}
            self.logger.info(f"\nFeature set: {feature_name} (shape: {X_tr.shape})")
            
            for clf_name, classifier in classifiers.items():
                try:
                    result = self.train_and_evaluate_classifier(
                        X_tr, X_te, self.y_train, self.y_test, 
                        f"{feature_name}-{clf_name}", classifier
                    )
                    results[feature_name][clf_name] = result
                    self.experiment_step += 1
                except Exception as e:
                    self.logger.error(f"Error with {feature_name}-{clf_name}: {str(e)}")
                    results[feature_name][clf_name] = None
        
        self.results = results
        
        # Save results to text file
        self.save_results_to_file()
        
        return results
    
    def create_classifiers(self):
        """Create classifiers from Hydra configuration."""
        cfg = self.cfg.classifiers
        
        classifiers = {}
        
        if hasattr(cfg, 'random_forest'):
            rf_cfg = cfg.random_forest
            classifiers['Random Forest'] = RandomForestClassifier(
                n_estimators=rf_cfg.n_estimators,
                max_depth=rf_cfg.max_depth,
                min_samples_split=rf_cfg.min_samples_split,
                min_samples_leaf=rf_cfg.min_samples_leaf,
                random_state=self.cfg.experiment.seed
            )
        
        if hasattr(cfg, 'svm'):
            svm_cfg = cfg.svm
            classifiers['SVM'] = SVC(
                kernel=svm_cfg.kernel,
                C=svm_cfg.C,
                gamma=svm_cfg.gamma,
                random_state=self.cfg.experiment.seed
            )
        
        if hasattr(cfg, 'knn'):
            knn_cfg = cfg.knn
            classifiers['KNN'] = KNeighborsClassifier(
                n_neighbors=knn_cfg.n_neighbors,
                weights=knn_cfg.weights
            )
        
        if hasattr(cfg, 'naive_bayes'):
            nb_cfg = cfg.naive_bayes
            classifiers['Naive Bayes'] = GaussianNB(
                var_smoothing=nb_cfg.var_smoothing
            )
        
        if hasattr(cfg, 'neural_network'):
            nn_cfg = cfg.neural_network
            classifiers['Neural Network'] = MLPClassifier(
                hidden_layer_sizes=tuple(nn_cfg.hidden_layer_sizes),
                max_iter=nn_cfg.max_iter,
                learning_rate_init=nn_cfg.learning_rate_init,
                alpha=nn_cfg.alpha,
                random_state=self.cfg.experiment.seed
            )
        
        return classifiers
    
    def save_results_to_file(self):
        """Save detailed results to a text file."""
        if not self.results:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.cfg.logging.log_dir, f"results_{timestamp}.txt")
        
        with open(results_file, 'w') as f:
            f.write("GMRA CLASSIFICATION EXPERIMENT RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment: {self.cfg.experiment.name}\n")
            f.write(f"Dataset: {self.cfg.data.dataset.upper()}\n")
            f.write(f"Training samples: {self.X_train.shape[0]}\n")
            f.write(f"Test samples: {self.X_test.shape[0]}\n")
            f.write(f"Features: {self.X_train.shape[1]}\n")
            f.write(f"Random seed: {self.cfg.experiment.seed}\n\n")
            
            # Write configuration summary
            f.write("CONFIGURATION SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"GMRA leafsize: {self.cfg.gmra.leafsize}\n")
            f.write(f"GMRA min_points: {self.cfg.gmra.min_points}\n")
            f.write(f"PCA components: {self.cfg.pca.n_components}\n")
            f.write(f"Test size: {self.cfg.data.test_size}\n\n")
            
            # Write detailed results
            for feature_name, classifiers in self.results.items():
                f.write(f"Feature Type: {feature_name}\n")
                f.write("-" * 40 + "\n")
                
                for clf_name, result in classifiers.items():
                    if result is not None:
                        f.write(f"  Classifier: {clf_name}\n")
                        f.write(f"    Accuracy: {result['accuracy']:.4f}\n")
                        f.write(f"    Precision: {result['precision']:.4f}\n")
                        f.write(f"    Recall: {result['recall']:.4f}\n")
                        f.write(f"    F1-Score: {result['f1']:.4f}\n")
                        f.write(f"    Training Time: {result['training_time']:.4f}s\n")
                        f.write(f"    Prediction Time: {result['prediction_time']:.4f}s\n")
                        f.write("\n")
                    else:
                        f.write(f"  Classifier: {clf_name} - FAILED\n\n")
                f.write("\n")
            
            # Best results summary
            f.write("SUMMARY\n")
            f.write("="*30 + "\n")
            
            data = []
            for feature_name, classifiers in self.results.items():
                for clf_name, result in classifiers.items():
                    if result is not None:
                        data.append({
                            'Feature': feature_name,
                            'Classifier': clf_name,
                            'Accuracy': result['accuracy']
                        })
            
            if data:
                df = pd.DataFrame(data)
                best_overall = df.loc[df['Accuracy'].idxmax()]
                f.write(f"Best Overall: {best_overall['Accuracy']:.4f} ({best_overall['Feature']} + {best_overall['Classifier']})\n")
                
                for feature_name in df['Feature'].unique():
                    feature_data = df[df['Feature'] == feature_name]
                    best_acc = feature_data.loc[feature_data['Accuracy'].idxmax()]
                    f.write(f"Best {feature_name}: {best_acc['Accuracy']:.4f} ({best_acc['Classifier']})\n")
        
        self.logger.info(f"Results saved to: {results_file}")
    
    def close_tensorboard(self):
        """Close TensorBoard writer."""
        if self.tb_writer:
            self.tb_writer.close()
            self.logger.info("TensorBoard writer closed")
    
    def plot_results(self):
        """Plot experiment results."""
        if not self.results:
            self.logger.error("No results to plot. Run experiment first.")
            return
        
        # Create results DataFrame
        data = []
        for feature_name, classifiers in self.results.items():
            for clf_name, result in classifiers.items():
                if result is not None:
                    data.append({
                        'Feature': feature_name,
                        'Classifier': clf_name,
                        'Accuracy': result['accuracy'],
                        'F1 Score': result['f1'],
                        'Training Time': result['training_time'],
                        'Prediction Time': result['prediction_time']
                    })
        
        df = pd.DataFrame(data)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        accuracy_pivot = df.pivot(index='Classifier', columns='Feature', values='Accuracy')
        sns.heatmap(accuracy_pivot, annot=True, fmt='.4f', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_xlabel('Feature Type')
        axes[0, 0].set_ylabel('Classifier')
        
        # F1 Score comparison
        f1_pivot = df.pivot(index='Classifier', columns='Feature', values='F1 Score')
        sns.heatmap(f1_pivot, annot=True, fmt='.4f', cmap='Greens', ax=axes[0, 1])
        axes[0, 1].set_title('F1 Score Comparison')
        axes[0, 1].set_xlabel('Feature Type')
        axes[0, 1].set_ylabel('Classifier')
        
        # Training time comparison
        sns.barplot(data=df, x='Feature', y='Training Time', hue='Classifier', ax=axes[1, 0])
        axes[1, 0].set_title('Training Time Comparison')
        axes[1, 0].set_xlabel('Feature Type')
        axes[1, 0].set_ylabel('Training Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Prediction time comparison
        sns.barplot(data=df, x='Feature', y='Prediction Time', hue='Classifier', ax=axes[1, 1])
        axes[1, 1].set_title('Prediction Time Comparison')
        axes[1, 1].set_xlabel('Feature Type')
        axes[1, 1].set_ylabel('Prediction Time (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot if configured
        if self.cfg.logging.save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = os.path.join(self.cfg.logging.log_dir, f"results_plot_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to: {plot_file}")
        
        plt.show()
        
        # Print summary
        self.logger.info("\nEXPERIMENT SUMMARY")
        self.logger.info("="*50)
        
        # Best accuracy for each feature type
        for feature_name in df['Feature'].unique():
            feature_data = df[df['Feature'] == feature_name]
            best_acc = feature_data.loc[feature_data['Accuracy'].idxmax()]
            self.logger.info(f"{feature_name}: Best accuracy = {best_acc['Accuracy']:.4f} ({best_acc['Classifier']})")
        
        # Overall best performance
        best_overall = df.loc[df['Accuracy'].idxmax()]
        self.logger.info(f"\nOverall best: {best_overall['Accuracy']:.4f} ({best_overall['Feature']} + {best_overall['Classifier']})")
        
        return df

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to run the classification experiment with Hydra."""
    # Print configuration
    print("="*60)
    print("GMRA CLASSIFICATION EXPERIMENT - HYDRA CONFIG")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))
    print("="*60)
    
    # Create and run experiment
    experiment = GMRAClassificationExperiment(cfg)
    
    try:
        results = experiment.run_classification_experiment()
        
        # Plot results
        experiment.plot_results()
        
        experiment.logger.info("Experiment completed successfully!")
        
    except Exception as e:
        experiment.logger.error(f"Experiment failed: {str(e)}")
        raise
    finally:
        # Close TensorBoard writer
        experiment.close_tensorboard()

if __name__ == "__main__":
    main()
