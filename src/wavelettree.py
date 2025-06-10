# SYSTEM IMPORTS
from typing import List, Tuple, Union
from scipy.linalg import qr
import numpy as np


# PYTHON PROJECT IMPORTS
import os
from .helpers import node_function, rand_pca
from .utils import *

WaveletNodeType = "WaveLetNode"

class WaveletNode(object):
    def __init__(self,
                 idxs: np.ndarray,
                 X: np.ndarray,
                 manifold_dim: int,
                 max_dim: int,
                 is_leaf: bool,
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
        self.node_j = node_j
        self.node_k = node_k


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
            threshold=threshold,
            precision=precision, 
            inverse=inverse
        )

        self.wav_basis: np.ndarray = None
        self.wav_sigmas: np.ndarray = None
        self.wav_consts: np.ndarray = None

        self.CelWavCoeffs = {}

    def make_transform(self,
                       X: np.ndarray,
                       manifold_dim: int,
                       max_dim: int,
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
                 thresholds: Union[float, np.ndarray] = 0.5,
                 precisions: Union[float, np.ndarray] = 1e-2,
                 inverse = False) -> None:

        '''
        init the tree with parameters: manifold_dim, max_dim, threshold, precision 
        inverse = True if dataset is in shape [d,n]
        '''

        if not isinstance(manifold_dims, np.ndarray):
            self.manifold_dims = np.ones(dyadic_tree.height, dtype=int)* int(manifold_dims)
        if not isinstance(thresholds, np.ndarray):
            self.thresholds = np.ones(dyadic_tree.height, dtype=float) * thresholds
        if not isinstance(precisions, np.ndarray):
            self.precisions = np.ones(dyadic_tree.height, dtype=float) * precisions

        self.max_dim: int = max_dim

        self.height: int = dyadic_tree.height

        self.dyadic_tree = dyadic_tree
        self.root: WaveletNode = None

        self.inverse = inverse

        self.j_k_to_wavelet_node = {}

        print('info: computing basis and wavelets for dyadic tree of height', dyadic_tree.height)
        print('info: this may take time..')
        
        self.make_basis(dyadic_tree, X)
        self.make_wavelets(X)
       
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
            threshold       = self.thresholds[0],
            precision       = self.precisions[0],
            inverse         = self.inverse,
            node_j          = 0,
            node_k          = 0
        )
        cell_root.node_j = 0
        cell_root.node_k = 0
        self.j_k_to_wavelet_node[(0, 0)] = self.root
                                

        current_cells = [cell_root]
        current_nodes = [self.root]

        for level in range(1, dyadic_tree.height+1):
            next_cells = list()
            next_nodes = list()

            k_counter = 0 

            for cell, node in zip(current_cells, current_nodes):
                # print('info: making wavelet node for cell', cell.idxs, 'at level', level)
                # print('info: cell have children:', len(cell.children))
                for child_idx, child_cell in enumerate(cell.children):
                    new_node            = WaveletNode(
                        idxs            = np.sort(child_cell.idxs),
                        X               = X,
                        manifold_dim    = self.manifold_dims[level],
                        max_dim         = self.max_dim,
                        is_leaf         = len(child_cell.children) == 0,
                        threshold       = self.thresholds[level],
                        precision       = self.precisions[level],
                        inverse         = self.inverse,
                        node_j          = level,
                        node_k          = k_counter
                    )

                    child_cell.node_j = level
                    child_cell.node_k = k_counter

                    self.j_k_to_wavelet_node[(level, k_counter)] = new_node
                    k_counter += 1

                    new_node.parent = node
                    node.children.append(new_node)

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
            # print('info: making transforms at level', j)    
            nodes = nodes_at_layers[j]
            # print("layer %s" % j)
            for node in nodes:
                node.make_transform(X.T, self.manifold_dims[j], self.max_dim,
                                     self.thresholds[j], self.precisions[j])

    
    def query_leaf(self, X: np.ndarray):
        '''
        Query the leaf nodes for the given data points
        X: data points, [n, d]
        Returns:
        leafs: list of WaveletNodeType objects corresponding to the leaf nodes

        
        The pipe line when a new data point is like this:
        Say we have x, we need to find the dyadic tree leaf that is closest to x
        and then need to find the wavelet node that corresponds to that leaf.

        '''
        leafs = []
        cover_leafs = self.dyadic_tree.query_leaf(X)
        wavelet_leafs = [self.j_k_to_wavelet_node[(leaf.node_j, leaf.node_k)] for leaf in cover_leafs]
        
        return wavelet_leafs

    def fgwt(self, X):
        '''
        Compute the forward gmra wavelet transform
        X: data points, [n, d]
        Returns:
        Qjx: list of lists, where each list corresponds to a data point
        and contains the wavelet coefficients at each level of the tree.

        This implement the algorithm in Figure 3.
        '''
        leafs    = self.query_leaf(X)
        leafs_jk= [(leaf.node_j, leaf.node_k) for leaf in leafs]
        Qjx = [None] * X.shape[0]

        for idx, leaf in enumerate(leafs):
            x = X[idx].reshape(1, -1)  #  a row
            pjx = leaf.basis @ (x.T-leaf.center) 
            qjx = leaf.wav_basis  @ leaf.basis.T @ pjx

            Qjx[idx]=[qjx]

            pJx = pjx
            
            p = path(leaf)

            for n in reversed(p[1:-1]):
                pjx = n.basis @ leaf.basis.T @ pJx + \
                        n.basis @ ( leaf.center - n.center ) 
                qjx = n.wav_basis @ n.basis.T @ pjx
                Qjx[idx].append(qjx)

            n = p[0]
            pjx = n.basis @ leaf.basis.T @ pJx + n.basis @ ( leaf.center - n.center ) 
            qjx = pjx
            Qjx[idx].append(qjx)
            Qjx[idx] = list(reversed(Qjx[idx]))
        return Qjx, leafs_jk

    def igwt(self, gmra_q_coeff, leaves_j_k ,shape):
        '''
        Compute the inverse gmra wavelet transform
        gmra_q_coeff: list of lists, where each list corresponds to a data point
        and contains the wavelet coefficients at each level of the tree.
        shape: shape of the original data points, [n, d]
        Implement the algorithm in Figure 4.
        '''
        X_recon = np.zeros(shape, dtype=np.float64)

        # iterate over data points
        for i in range(len(gmra_q_coeff)):
            # coefficient and leaf node for this data
            coeffs = list(reversed(gmra_q_coeff[i]))# leaf -> root
            leaf   = self.j_k_to_wavelet_node[leaves_j_k[i]]
            lvl_from_leaf = 0 

            # begin reconstruct
            # leaves level treat differently
            Qjx    = leaf.wav_basis.T @ coeffs[lvl_from_leaf] + leaf.wav_consts
            leaf   = leaf.parent
            lvl_from_leaf += 1

            while leaf.parent is not None:
                Qjx +=( leaf.wav_basis.T @ coeffs[lvl_from_leaf] + leaf.wav_consts +
                        leaf.parent.basis.T @ leaf.parent.basis @ Qjx)
                leaf = leaf.parent
                lvl_from_leaf += 1
            
            # root level also treat differently
            Qjx += leaf.basis.T @ coeffs[lvl_from_leaf] + leaf.center

            X_recon[i:i+1,:] = Qjx.T
        return X_recon
                    
        # old implementation:                
        # for leaf in get_leafs(wavelet_tree.root):
        #     data_idx  = leaf.idxs[0]
        #     # print('reconstructing', data_idx)
        #     # idx_reconstructed.append(data_idx)
        #     # print('data_idx', data_idx)
        #     chain = path(leaf)
            
        #     ct=-1

        #     Qjx = leaf.wav_basis.T @ gmra_q_coeff[data_idx][ct] + leaf.wav_consts

        #     new_chain = chain[1:-1]
        #     for jj, n in reversed(list(enumerate(new_chain))):
        #         # print(len(gmra_q_coeff[data_idx]))
        #         ct-=1
        #         Qjx += (n.wav_basis.T @ gmra_q_coeff[data_idx][ct] + n.wav_consts +
        #                 new_chain[jj-1].basis.T @ new_chain[jj-1].basis @ Qjx)
        #         # print(ct)
        #     ct-=1
        #     Qjx += chain[0].basis.T@ gmra_q_coeff[data_idx][ct] + chain[0].center 
        #     X_recon[data_idx:data_idx+1,:] = Qjx.T
        # return X_recon