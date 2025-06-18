import numpy as np
from typing import List, Tuple, Union
from scipy.linalg import qr
import os
import logging
from .helpers import node_function, rand_pca
from .utils import *
from .dyadictreenode import DyadicTreeNode

# our own implementation of DyadicTree using Python
# DyadicTree will be constructed from the CoverTree.
# root node will be at CoverTree max scale (root node).

def get_idx_sublevel(node):
    """
    Get all the index of the nodes under node of a CoverTreeNode
    :param node: node
    :return: numpy 1d array of indices
    """
    if hasattr(node, 'idx'):
        return node.idx
    else:
        idxs = [get_idx_sublevel(child) for child in node.children]
        if idxs:
            return np.concatenate(idxs)
        else:
            return np.array([], dtype=int)

class DyadicTree:
    def __init__(self, cover_tree, X=None, manifold_dims=None, max_dim=None, 
                 thresholds=0.5, precisions=1e-2, inverse=False):
        if cover_tree.n == 0:
            raise ValueError("Cover tree is empty")
        self.root = DyadicTreeNode(get_idx_sublevel(cover_tree.root), parent=None)
        self.height = 1

        # underlying cover tree
        self.cover_tree = cover_tree
        self.idx_to_leaf_node = {}

        # Wavelet-related attributes
        self.j_k_to_node = {}
        self.inverse = inverse
        
        # Scikit-learn style attributes
        self._is_fitted = False
        self._X_shape = None
        
        self._setup_wavelet_params(manifold_dims, max_dim, thresholds, precisions)
            
        self.build_tree(self.root, cover_tree.root)
        
        if X is not None and manifold_dims is not None and max_dim is not None:
            self.fit(X)

    def _setup_wavelet_params(self, manifold_dims, max_dim, thresholds, precisions):
        if manifold_dims is not None:
            if not isinstance(manifold_dims, np.ndarray):
                self.manifold_dims = np.ones(self.height, dtype=int) * int(manifold_dims)
            else:
                self.manifold_dims = manifold_dims
        if thresholds is not None:
            if not isinstance(thresholds, np.ndarray):
                self.thresholds = np.ones(self.height, dtype=float) * thresholds
            else:
                self.thresholds = thresholds
        if precisions is not None:
            if not isinstance(precisions, np.ndarray):
                self.precisions = np.ones(self.height, dtype=float) * precisions
            else:
                self.precisions = precisions
        self.max_dim = max_dim

    def build_tree(self, node, cover_node, level=1):
        """
        Recursively build the DyadicTree from the CoverTree. 
        Remember to update idx_to_leaf_node mapping.
        """
        logging.debug(f"Building tree at level {level}, node indices: {get_idx_sublevel(cover_node)}")

        if level >= self.height:
            self.height = level+1
            logging.debug(f"Updated tree height to {self.height}")

        if hasattr(cover_node, 'idx'):
            # the leaf node will have idx
            child_node = DyadicTreeNode(cover_node.idx, parent=node)
            logging.debug(f"Created leaf node at level {level} with indices: {cover_node.idx}")

            self.idx_to_leaf_node[cover_node.idx[0]] = child_node

            node.add_child(child_node)
        else:
            logging.debug(f"Processing internal node at level {level} with {len(cover_node.children)} children")
            for i, child in enumerate(cover_node.children):
                child_node = DyadicTreeNode(get_idx_sublevel(child), parent=node)
                logging.debug(f"Created child {i+1}/{len(cover_node.children)} at level {level}")
                node.add_child(child_node)
                self.build_tree(child_node, child, level + 1)

    def traverse(self, only_print_level=None):
        def traverse_from_node(node, level=0, only_print_level=None):
            """
            Print the tree, starting from the root node.
            At every level, use a single dash as level indicator.
            eg: root node has '-', child node has '--', etc.
            If only_print_level is set, only print nodes at that level.

            """
            if only_print_level is None or level == only_print_level:
                print("-" * (level + 1), node.idxs)
            for child in node.children:
                traverse_from_node(child, level + 1, only_print_level)

        traverse_from_node(self.root, level=0, only_print_level=only_print_level)
    
    def plot_tree(self):
        '''
        plot the tree using matplotlib
        '''
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        def plot_node(node, x, y, ax, level=0):
            """
            Plot the node and its children.
            """
            color = 'red' if getattr(node, 'fake_node', False) else 'black'
            ax.text(x, y, str(node.idxs), ha='center', va='center', fontsize=8, color=color)
            for i, child in enumerate(node.children):
                child_x = x + (i - len(node.children) / 2) * 0.1
                child_y = y - 0.1
                ax.plot([x, child_x], [y, child_y], 'k-')
                plot_node(child, child_x, child_y, ax, level + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Structure')
        ax.axis('off')
        plot_node(self.root, 0, 0, ax)
        plt.show()

    def grow_tree(self):
        """
        Grow tree so that every leave have the same levels = tree height.
        if a leave is not enough height, we duplicate node until it reaches the height of the tree.
        set the node.fake_node = True
        """
        
        def grow_node(node, current_level):
            if current_level+1 >= self.height:
                return

            if len(node.children) == 0:
                # duplicate node
                new_node = DyadicTreeNode(node.idxs, parent=node)
                new_node.fake_node = True

                # update the idx to leaf node mapping with the new  node
                self.idx_to_leaf_node[node.idxs[0]] = new_node
                #!#

                # register and grow
                node.add_child(new_node)
                grow_node(new_node, current_level + 1)
            else:
                for child in node.children:
                    grow_node(child, current_level + 1)

        grow_node(self.root, 0)

        return self
    
    def query_leaf(self, X):
        """
        Query for the leaf that is closest to the data point using self.cover_tree
        use the underlying cover tree
        _,nn_idx = python_covertree.query(X_query)
        then query node using self.idx_to_leaf_node[nn_idx]
        X : numpy array of shape (n_samples, n_features)
        """

        _, nn_idx = self.cover_tree.query(X)
        leaf_nodes = [self.idx_to_leaf_node[idx] for idx in nn_idx]
        return leaf_nodes

    def make_basis(self, X: np.ndarray) -> None:
        '''
        Recursively compute basis for all nodes in the tree
        '''
        logging.info("Starting basis construction for DyadicTree")
        
        # Setup root node
        self.root.is_leaf = len(self.root.children) == 0
        self.root.node_j = 0
        self.root.node_k = 0
        
        # Compute basis for root using node's make_basis method
        self.root.make_basis(X, self.manifold_dims[0], self.max_dim,
                            self.thresholds[0], self.precisions[0], self.inverse)
        
        self.j_k_to_node[(0, 0)] = self.root
        
        # Recursively process all levels
        self._make_basis_recursive(self.root, X, level=0)

    def _make_basis_recursive(self, node: DyadicTreeNode, X: np.ndarray, level: int) -> None:
        '''
        Recursive helper for make_basis
        '''
        if level + 1 >= self.height:
            return
            
        logging.debug(f"Processing level {level+1} with {len(node.children)} children")
        
        k_counter = 0
        if level == 0:  # Start counting from level 1
            k_counter = 0
        else:
            # Find the current k_counter for this level
            existing_nodes_at_level = [n for (j, k), n in self.j_k_to_node.items() if j == level + 1]
            k_counter = len(existing_nodes_at_level)
        
        for child in node.children:
            child.is_leaf = len(child.children) == 0
            child.node_j = level + 1
            child.node_k = k_counter
            
            # Compute basis for child using node's make_basis method
            child.make_basis(X, 
                           self.manifold_dims[level + 1] if level + 1 < len(self.manifold_dims) else self.manifold_dims[-1],
                           self.max_dim,
                           self.thresholds[level + 1] if level + 1 < len(self.thresholds) else self.thresholds[-1],
                           self.precisions[level + 1] if level + 1 < len(self.precisions) else self.precisions[-1],
                           self.inverse)

            self.j_k_to_node[(level + 1, k_counter)] = child
            k_counter += 1
            
            # Recursively process children
            self._make_basis_recursive(child, X, level + 1)

    def make_wavelets(self, X: np.ndarray) -> None:
        print('info: making wavelets')
        logging.debug("Starting wavelet construction")
        
        nodes_at_layers = [[self.root]]
        current_layer = nodes_at_layers[0]
        for level in range(1, self.height):
            next_layer = list()
            for node in current_layer:
                for child in node.children:
                    next_layer.append(child)
            nodes_at_layers.append(next_layer)
            current_layer = next_layer
            logging.debug(f"Layer {level} has {len(next_layer)} nodes")

        for j in range(self.height-1, -1, -1):
            nodes = nodes_at_layers[j]
            logging.debug(f"Processing wavelets for level {j} with {len(nodes)} nodes")
            for i, node in enumerate(nodes):
                logging.debug(f"Making transform for node {i+1}/{len(nodes)} at level {j}")
                node.make_transform(X.T, 
                                  self.manifold_dims[j] if j < len(self.manifold_dims) else self.manifold_dims[-1], 
                                  self.max_dim,
                                  self.thresholds[j] if j < len(self.thresholds) else self.thresholds[-1], 
                                  self.precisions[j] if j < len(self.precisions) else self.precisions[-1])
    
    def fgwt_all_node(self, X):
        '''
        Compute the forward gmra wavelet transform for all nodes in the tree.
        fgwt is only depends on computation from the leaf has x, traversing the tree to the root.
        fgwt_all_node will compute basically the same thing, but for node that not in the path, it's 0
        '''
        logging.debug("Starting forward GMRA wavelet transform for all nodes")
        
        # for each node, determine the dimension of coefficients.
        # the final dimension of coefficients is the sum of all dimensions of the nodes.

        final_dim = 0 
        
        
        



    def fgwt(self, X):
        '''
        Compute the forward gmra wavelet transform
        '''
        logging.debug(f"Starting forward GMRA wavelet transform for {X.shape[0]} data points")
        
        leafs = self.query_leaf(X)
        leafs_jk = [(leaf.node_j, leaf.node_k) for leaf in leafs]
        
        logging.debug(f"Found {len(leafs)} leaf nodes, levels range: j={min(jk[0] for jk in leafs_jk)} to j={max(jk[0] for jk in leafs_jk)}")
        
        Qjx = [None] * X.shape[0]

        for idx, leaf in enumerate(leafs):
            if idx % 20 == 0:  # Log every 20th point to avoid spam
                logging.debug(f"Processing point {idx+1}/{len(leafs)}, leaf at (j={leaf.node_j}, k={leaf.node_k})")
            
            x = X[idx].reshape(1, -1)  #  a row
            pjx = leaf.basis @ (x.T-leaf.center) 
            qjx = leaf.wav_basis @ leaf.basis.T @ pjx

            Qjx[idx]=[qjx]
            pJx = pjx
            
            p = path(leaf)
            logging.debug(f"Point {idx}: path length {len(p)}, leaf->root traversal")

            for n in reversed(p[1:-1]):
                pjx = n.basis @ leaf.basis.T @ pJx + \
                        n.basis @ ( leaf.center - n.center ) 
                qjx = n.wav_basis @ n.basis.T @ pjx
                Qjx[idx].append(qjx)
                logging.debug(f"Point {idx}: processed node at (j={n.node_j}, k={n.node_k}), qjx shape: {qjx.shape}")

            n = p[0]
            pjx = n.basis @ leaf.basis.T @ pJx + n.basis @ ( leaf.center - n.center ) 
            qjx = pjx
            Qjx[idx].append(qjx)
            Qjx[idx] = list(reversed(Qjx[idx]))
            
            logging.debug(f"Point {idx}: completed, total coefficients at {len(Qjx[idx])} levels")
        
        logging.debug("Forward GMRA wavelet transform completed")
        return Qjx, leafs_jk

    def igwt(self, gmra_q_coeff, leaves_j_k, shape):
        '''
        Compute the inverse gmra wavelet transform
        '''
        logging.debug(f"Starting inverse GMRA wavelet transform for {len(gmra_q_coeff)} data points")
        logging.debug(f"Reconstruction target shape: {shape}")
        
        X_recon = np.zeros(shape, dtype=np.float64)

        # iterate over data points
        for i in range(len(gmra_q_coeff)):
            if i % 20 == 0:  # Log every 20th point to avoid spam
                logging.debug(f"Reconstructing point {i+1}/{len(gmra_q_coeff)}")
            
            # coefficient and leaf node for this data
            coeffs = list(reversed(gmra_q_coeff[i]))# leaf -> root
            leaf = self.j_k_to_node[leaves_j_k[i]]
            lvl_from_leaf = 0
            
            logging.debug(f"Point {i}: starting from leaf (j={leaf.node_j}, k={leaf.node_k}), {len(coeffs)} coefficient levels")

            # begin reconstruct
            # leaves level treat differently
            Qjx = leaf.wav_basis.T @ coeffs[lvl_from_leaf] + leaf.wav_consts
            logging.debug(f"Point {i}: leaf reconstruction, Qjx shape: {Qjx.shape}")
            
            leaf = leaf.parent
            lvl_from_leaf += 1

            while leaf.parent is not None:
                Qjx += (leaf.wav_basis.T @ coeffs[lvl_from_leaf] + leaf.wav_consts +
                        leaf.parent.basis.T @ leaf.parent.basis @ Qjx)
                logging.debug(f"Point {i}: intermediate level (j={leaf.node_j}, k={leaf.node_k}), Qjx shape: {Qjx.shape}")
                leaf = leaf.parent
                lvl_from_leaf += 1
            
            # root level also treat differently
            Qjx += leaf.basis.T @ coeffs[lvl_from_leaf] + leaf.center
            logging.debug(f"Point {i}: root level reconstruction, final Qjx shape: {Qjx.shape}")

            X_recon[i:i+1,:] = Qjx.T
            
            if i % 20 == 0:
                recon_norm = np.linalg.norm(X_recon[i])
                logging.debug(f"Point {i}: reconstruction norm: {recon_norm:.6f}")
        
        logging.debug("Inverse GMRA wavelet transform completed")
        return X_recon

    # ========== Scikit-learn Style API ==========
    
    def fit(self, X):
        """
        Learn the basis and wavelet basis from the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        
        self._X_shape = X.shape
        
        # Learn basis and wavelets
        self.make_basis(X)
        self.make_wavelets(X)
        
        self._is_fitted = True
        return self
    
    def fit_transform(self, X):
        """
        Fit the model with X and apply the forward GMRA wavelet transform to X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        Returns
        -------
        X_transformed : tuple
            Tuple containing (coefficients, leaf_indices) where:
            - coefficients: list of wavelet coefficients for each data point
            - leaf_indices: list of (j, k) tuples indicating leaf node for each point
        """
        return self.fit(X).transform(X)
    
    def transform(self, X):
        """
        Apply the forward GMRA wavelet transform to X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : tuple
            Tuple containing (coefficients, leaf_indices) where:
            - coefficients: list of wavelet coefficients for each data point
            - leaf_indices: list of (j, k) tuples indicating leaf node for each point
        """
        if not self._is_fitted:
            raise ValueError("This DyadicTree instance is not fitted yet. "
                           "Call 'fit' with appropriate arguments before using this estimator.")
        
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        
        if X.shape[1] != self._X_shape[1]:
            raise ValueError(f"X has {X.shape[1]} features, but DyadicTree is expecting "
                           f"{self._X_shape[1]} features as seen in fit.")
        
        return self.fgwt(X)
    
    def inverse_transform(self, X_transformed):
        """
        Apply the inverse GMRA wavelet transform to reconstruct the original data.
        
        Parameters
        ----------
        X_transformed : tuple
            Tuple containing (coefficients, leaf_indices) as returned by transform().
            
        Returns
        -------
        X_reconstructed : array of shape (n_samples, n_features)
            Reconstructed data.
        """
        if not self._is_fitted:
            raise ValueError("This DyadicTree instance is not fitted yet. "
                           "Call 'fit' with appropriate arguments before using this estimator.")
        
        if not isinstance(X_transformed, tuple) or len(X_transformed) != 2:
            raise ValueError("X_transformed must be a tuple of (coefficients, leaf_indices) "
                           "as returned by transform().")
        
        coefficients, leaf_indices = X_transformed
        
        if len(coefficients) != len(leaf_indices):
            raise ValueError("Length of coefficients and leaf_indices must match.")
        
        # Determine the shape for reconstruction
        n_samples = len(coefficients)
        n_features = self._X_shape[1]
        shape = (n_samples, n_features)
        
        return self.igwt(coefficients, leaf_indices, shape)
    
    # ========== Additional utility methods ==========
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'cover_tree': self.cover_tree,
            'manifold_dims': getattr(self, 'manifold_dims', None),
            'max_dim': self.max_dim,
            'thresholds': getattr(self, 'thresholds', None),
            'precisions': getattr(self, 'precisions', None),
            'inverse': self.inverse
        }
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
            
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self

    # ========== Original methods ==========




