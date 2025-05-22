# SYSTEM IMPORTS
from typing import List, Tuple, Union
from scipy.linalg import qr
import numpy as np


# PYTHON PROJECT IMPORTS
import os
from .helpers import node_function, rand_pca

WaveletNodeType = "WaveLetNode"

class WaveletNode(object):
    def __init__(self,
                 idxs: np.ndarray,
                 X: np.ndarray,
                 manifold_dim: int,
                 max_dim: int,
                 is_leaf: bool,
                 shelf = None,
                 threshold: float = 0.5,
                 precision: float = 1e-2, inverse = False,
                 node_j=-1, node_k=-1) -> None:
        self.idxs: np.ndarray = idxs
        self.is_leaf: bool = is_leaf
        self.children: List[WaveletNodeType] = list() # type: ignore
        self.parent: WaveletNodeType = None # type: ignore

        # center is the mean of data points
        # size is essentially x.shape[0] # dim
        # radius is the max distance from the center
        # basis is the pca basis
        # sigmas is the pca singular values
        # print('computing basis for cell j,k:', node_j, node_k, 'with size:', self.idxs.shape,'pts')
        self.node_j= node_j
        self.node_k=node_k


        #self.Z only for debug purpose
        # basis: [d_manifold, d_original]
        # sigmas: [d_manifold]
        # Z: [d_manifold, n_ponint]
        # so X_hat = basis.T @ np.diag(sigmas) @ Z + center

        self.center, self.size, self.radius, self.basis, self.sigmas, self.Z = node_function(
            np.atleast_2d(X[idxs,:]),
            manifold_dim,
            max_dim,
            is_leaf,
            shelf=shelf,
            threshold=threshold,
            precision=precision, 
            inverse=inverse
        )

        # debug only, make sure that X[idxs,:] is the same as pca reconstruct
        # x_orig = np.atleast_2d(X[idxs,:])
        # multiply with sqrt of size because for somereason the scale it in pca
        # x_hat = np.sqrt(self.size) *  self.basis.T @ np.diag(self.sigmas[:len(self.sigmas)-1])[:self.basis.T.shape[1]] @ self.Z + self.center


        # print('difference:', np.linalg.norm(x_orig.T - x_hat))  
        # assert np.linalg.norm(x_orig.T - x_hat) < .1, 'reconstruct fail'
        # print('number of basis', self.basis.shape[0])
        # print('reconstruction error:', np.linalg.norm(X[idxs,:] - x_hat))

        # self.center, self.size, self.radius, self.basis, self.sigmas, self.Z = no(
        #     np.atleast_2d(X[idxs,:]),
        #     manifold_dim,
        #     max_dim,
        #     is_leaf,
        #     shelf=shelf,
        #     threshold=threshold,
        #     precision=precision
        # )


        self.wav_basis: np.ndarray = None
        self.wav_sigmas: np.ndarray = None
        self.wav_consts: np.ndarray = None

        self.CelWavCoeffs = {}

    def make_transform(self,
                       X: np.ndarray,
                       manifold_dim: int,
                       max_dim: int,
                       shelf = None,
                       threshold: float = 0.5,
                       precision: float = 1e-2) -> None:
        '''
        X: (d,n)
        
        '''
        # from construct_localGeometricWavelets.m

        # print('info: making transform for node', self.node_j, self.node_k)
        # parent_basis: np.ndarray = self.basis
        Phijx = self.basis

        # print(self.idxs.shape, self.basis.shape)

        # if np.prod(parent_basis.shape) > 1:
        # wav_dims: np.ndarray = np.zeros(len(self.children))

        for i, c in enumerate(self.children):
            # If the child is a leaf node, there is no need to populate its wav vars, it doesnt have a wavelet
            # if len(c.idxs) ==1 or c.basis.shape[0] == 1:
            #     continue
            if np.prod(c.basis.shape) >= 1:

                # this is (I-Pjx)V_{j+1,x}
                # = (I - Phi_{jx}Phi_{jx}^T)V_{j+1,x}
                # = V_{j+1,x} - Phi_{jx}Phi_{jx}^T V_{j+1,x}
                # = Phi_{j,x} - Phi_{jx}Phi_{jx}^T Phi_{j+1,x}
                # but the basis is transposed so we do
                # = Phi_{j,x} - Phi_{j+1,x}Phi_{jx}^TPhi_{j,x}
                Phij1x = c.basis  # phi{j+1,x}
                # Phijx  = parent_basis  # phi{j,x}
                
                Y: np.ndarry = Phij1x - Phij1x @ Phijx.T @ Phijx
                # Y: d x D
            
                # Y: np.ndarray = c.basis-(c.basis.dot(parent_basis.T)).dot(parent_basis)

                # U, s, V = rand_pca(Y, min(min(X.shape), max_dim), inverse = True)
                # print('basis shape:',(c.basis.shape))
                # U, s, V = rand_pca(Y, min(min(X.shape), max_dim), inverse=True)
                U,s,_ = rand_pca(Y.T, min(min(X.shape), max_dim))
                wav_dims = (s > threshold).sum(dtype=np.int8)

                if wav_dims > 0:
                    # print('wave dims greater than 0 for node jk', self.node_j, self.node_k)
                    # c.wav_basis = V[:,:int(wav_dims[i])].T
                    c.wav_basis = U[:,:wav_dims].T # (nxd)
                    # print(c.wav_basis.shape)
                    c.wav_sigmas = s[:wav_dims]
                else:
                    c.wav_basis = np.zeros_like(U[:,:0].T)  # (nxd)
                    c.wav_sigmas = np.zeros_like(s[:0])


                # c.wav_consts = c.center - self.center # (2.14)
                # c.wav_consts = c.wav_consts - parent_basis.T.dot(parent_basis.dot(c.wav_consts)) #(2.15)
                tjx = c.center - self.center# (2.14)
                c.wav_consts = tjx - Phijx.T @ Phijx @ tjx
            else:
                c.wav_basis = np.zeros((X.shape[0], 0)).T # (nxd)
                c.wav_consts =  0 # np.zeros_like(c.center)
                # c.wav_sigmas =np.zeros()


class WaveletTree(object):
    def __init__(self,
                 dyadic_tree,
                 X: np.ndarray,
                 manifold_dims: Union[int, np.ndarray],
                 max_dim: int,
                 shelf = None,
                 thresholds: Union[float, np.ndarray] = 0.5,
                 precisions: Union[float, np.ndarray] = 1e-2,
                 inverse = False) -> None:

        '''
        init the tree with parameters: manifold_dim, max_dim, threshold, precision 
        inverse = True if dataset is in shape [d,n]
        '''

        if not isinstance(manifold_dims, np.ndarray):
            self.manifold_dims = np.ones(dyadic_tree.height, dtype=int)*\
                                 int(manifold_dims)
        if not isinstance(thresholds, np.ndarray):
            self.thresholds = np.ones(dyadic_tree.height, dtype=float) * thresholds
        if not isinstance(precisions, np.ndarray):
            self.precisions = np.ones(dyadic_tree.height, dtype=float) * precisions
        self.max_dim: int = max_dim
        self.shelf = shelf

        self.height: int = dyadic_tree.height
        self.root: WaveletNode = None
        self.num_nodes: int = 0
        self.inverse = inverse

        self.make_basis(dyadic_tree, X)

    def make_basis(self,
                   dyadic_tree,
                   X: np.ndarray) -> None:
        '''
        make basis
        X: data points, [n, d]
        dyadic_tree: DyadicTree
        
        This will create tree structure
        and for each node compute the pca basis and center
        '''

        print('info: making wavelet tree')
        cell_root = dyadic_tree.root
        self.root           = WaveletNode(
            idxs            = np.sort(cell_root.idxs),
            X               = X,
            manifold_dim    = self.manifold_dims[0],
            max_dim         = self.max_dim,
            is_leaf         = len(cell_root.children) == 0,
            shelf           = self.shelf,
            threshold       = self.thresholds[0],
            precision       = self.precisions[0],
            inverse         = self.inverse,
            node_j          = 0,
            node_k          = 0
        )
                                
        self.num_nodes += 1

        current_cells = [cell_root]
        current_nodes = [self.root]

        for level in range(1, dyadic_tree.height):
            next_cells = list()
            next_nodes = list()

            for cell, node in zip(current_cells, current_nodes):
                for child_idx, child_cell in enumerate(cell.children):
                    new_node            = WaveletNode(
                        idxs            = np.sort(child_cell.idxs),
                        X               = X,
                        manifold_dim    = self.manifold_dims[level],
                        max_dim         = self.max_dim,
                        is_leaf         = len(child_cell.children) == 0,
                        shelf           = self.shelf,
                        threshold       = self.thresholds[level],
                        precision       = self.precisions[level],
                        inverse         = self.inverse,
                        node_j          = level,
                        node_k          = child_idx
                    )

                    new_node.parent = node
                    node.children.append(new_node)
                    self.num_nodes += 1

                    next_cells.append(child_cell)
                    next_nodes.append(new_node)

            current_cells = next_cells
            current_nodes = next_nodes

    def make_wavelets(self,
                        X: np.ndarray) -> None:

        print('info: making wavelets')
        nodes_at_layers = [[self.root]]
        current_layer = nodes_at_layers[0]
        for level in range(1, self.height):
            next_layer = list()
            for node in current_layer:
                # print("level %s: basis.shape: %s" % (level, node.basis.shape))
                for child in node.children:
                    next_layer.append(child)

            nodes_at_layers.append(next_layer)
            current_layer = next_layer


        for j in range(self.height-1, -1, -1):
            # built transforms
            print('info: making transforms at level', j)    
            nodes = nodes_at_layers[j]
            # print("layer %s" % j)
            for node in nodes:
                node.make_transform(X.T, self.manifold_dims[j], self.max_dim,
                                    self.shelf, self.thresholds[j], self.precisions[j])
