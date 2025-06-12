import numpy as np
from typing import List, Tuple, Union
import logging
from .helpers import node_function, rand_pca

class DyadicTreeNode:
    def __init__(self, idxs, parent):
        self.idxs = idxs
        self.children = []
        self.parent = parent 
        self.fake_node = False 

        # Node identification
        self.node_k = None
        self.node_j = None
        self.is_leaf = False
        
        # Basis-related attributes
        self.center = None
        self.size = None
        self.radius = None
        self.basis = None
        self.sigmas = None
        self.Z = None  # Only used for debug, consider removing if not needed
        
        # Wavelet-related attributes  
        self.wav_basis = None
        self.wav_sigmas = None
        self.wav_consts = None
        # Remove CelWavCoeffs if not used

    def add_child(self, child_node):
        self.children.append(child_node)

    def __getitem__(self, index):
        return self.children[index]
    
    def __len__(self):
        return len(self.children)
    
    def make_basis(self, X: np.ndarray, manifold_dim: int, max_dim: int, 
                   threshold: float = 0.5, precision: float = 1e-2, 
                   inverse: bool = False) -> None:
        '''
        Compute basis for this node using node_function
        '''
        logging.debug(f"Computing basis for node (j={self.node_j}, k={self.node_k}) with {len(self.idxs)} points, is_leaf={self.is_leaf}")
        
        self.center, self.size, self.radius, self.basis, self.sigmas, self.Z = node_function(
            np.atleast_2d(X[self.idxs,:]),
            manifold_dim,
            max_dim,
            self.is_leaf,
            threshold=threshold,
            precision=precision, 
            inverse=inverse
        )
        
        logging.debug(f"Node (j={self.node_j}, k={self.node_k}) basis shape: {self.basis.shape}, sigmas: {len(self.sigmas)}")
    
    def make_transform(self, X: np.ndarray, manifold_dim: int, max_dim: int, 
                       threshold: float = 0.5, precision: float = 1e-2) -> None:
        '''
        Compute wavelet transform for this node's children
        X: (d,n)
        '''
        if not hasattr(self, 'basis') or self.basis is None:
            return
            
        Phijx = self.basis

        for child in self.children:
            if child.basis is None or np.prod(child.basis.shape) < 1:
                child.wav_basis = np.zeros((X.shape[0], 0)).T
                child.wav_consts = 0
                continue
                
            # Compute orthogonal component: (I-Pjx)V_{j+1,x}
            Phij1x = child.basis
            Y = Phij1x - Phij1x @ Phijx.T @ Phijx
        
            U, s, _ = rand_pca(Y.T, min(min(X.shape), max_dim))
            wav_dims = (s > threshold).sum(dtype=np.int8)

            if wav_dims > 0:
                child.wav_basis = U[:, :wav_dims].T
                child.wav_sigmas = s[:wav_dims]
            else:
                child.wav_basis = np.zeros((wav_dims, X.shape[0]))
                child.wav_sigmas = np.zeros(0)

            # Compute wavelet constants
            tjx = child.center - self.center
            child.wav_consts = tjx - Phijx.T @ Phijx @ tjx
