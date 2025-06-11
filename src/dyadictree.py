import numpy as np
from .helpers import node_function, rand_pca
from .utils import path

# our own implementation of DyadicTree and DyadicTreeNode using Python
# DyadicTreenNode will have point_indices point to indexes in dataset, and a list of children of type DyadicTreeNode
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

class DyadicTreeNode:
    def __init__(self, idxs, parent):
        self.idxs = idxs
        self.children = []
        self.parent = parent 
        self.fake_node = False 
        self.node_k = None
        self.node_j = None

        # Wavelet-related attributes
        self.center = None
        self.size = None
        self.radius = None
        self.basis = None
        self.sigmas = None
        self.Z = None
        self.wav_basis = None
        self.wav_sigmas = None
        self.wav_consts = None

    def add_child(self, child_node):
        self.children.append(child_node)

    def __getitem__(self, index):
        return self.children[index]
    
    def __len__(self):
        return len(self.children)
    
    def compute_basis(self, X, manifold_dim=0, max_dim=None, is_leaf=False, threshold=0.5, precision=1e-2, inverse=False):
        # Set max_dim default to X.shape[1] if not provided
        if max_dim is None:
            max_dim = X.shape[1]
        self.center, self.size, self.radius, self.basis, self.sigmas, self.Z = node_function(
            np.atleast_2d(X[self.idxs,:]),
            manifold_dim,
            max_dim,
            is_leaf,
            threshold=threshold,
            precision=precision, 
            inverse=inverse
        )

    def make_transform(self, X, manifold_dim=0, max_dim=None, threshold=0.5, precision=1e-2):
        # Set max_dim default to X.shape[1] if not provided
        if max_dim is None:
            max_dim = X.shape[1]
        Phijx = self.basis
        for c in self.children:
            if c.basis is not None and np.prod(c.basis.shape) >= 1:
                Phij1x = c.basis
                Y = Phij1x - Phij1x @ Phijx.T @ Phijx
                U, s, _ = rand_pca(Y.T, min(min(X.shape), max_dim))
                wav_dims = (s > threshold).sum(dtype=np.int8)
                if wav_dims > 0:
                    c.wav_basis = U[:,:wav_dims].T
                    c.wav_sigmas = s[:wav_dims]
                else:
                    c.wav_basis = np.zeros_like(U[:,:0].T)
                    c.wav_sigmas = np.zeros_like(s[:0])
                tjx = c.center - self.center
                c.wav_consts = tjx - Phijx.T @ Phijx @ tjx
            else:
                c.wav_basis = np.zeros((X.shape[0], 0)).T
                c.wav_consts = 0

class DyadicTree:
    def __init__(self, cover_tree):
        if cover_tree.n == 0:
            raise ValueError("Cover tree is empty")
        self.root = DyadicTreeNode(get_idx_sublevel(cover_tree.root), parent=None)
        self.height = 1

        # underlying cover tree
        self.cover_tree = cover_tree
        self.idx_to_leaf_node = {}

        # Add jk to node mapping
        self.jk_to_node = {}

        self.build_tree(self.root, cover_tree.root)

        # a idx to node mapping, every idx is essentially a point
        # we want to have idx -> node

        
    def build_tree(self, node, cover_node, level=1):
        """
        Recursively build the DyadicTree from the CoverTree. 
        Remember to update idx_to_leaf_node mapping.
        """
        # Assign node_j and node_k for jk mapping
        node.node_j = level - 1
        node.node_k = getattr(cover_node, 'ctr_idx', None) if hasattr(cover_node, 'ctr_idx') else None
        self.jk_to_node[(node.node_j, node.node_k)] = node

        if level+1 > self.height:
            self.height = level+1

        if hasattr(cover_node, 'idx'):
            # the leaf node will have idx
            child_node = DyadicTreeNode(cover_node.idx, parent=node)
            child_node.node_j = level
            child_node.node_k = getattr(cover_node, 'ctr_idx', None) if hasattr(cover_node, 'ctr_idx') else None
            self.jk_to_node[(child_node.node_j, child_node.node_k)] = child_node

            self.idx_to_leaf_node[cover_node.idx[0]] = child_node

            node.add_child(child_node)
        else:
            for child in cover_node.children:
                child_node = DyadicTreeNode(get_idx_sublevel(child), parent=node)
                child_node.node_j = level
                child_node.node_k = getattr(child, 'ctr_idx', None) if hasattr(child, 'ctr_idx') else None
                self.jk_to_node[(child_node.node_j, child_node.node_k)] = child_node

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

    def make_basis(self, X, manifold_dims, max_dim, thresholds, precisions, inverse=False):
        # Recursively compute basis for all nodes
        def _make_basis(node, level):
            is_leaf = len(node.children) == 0
            node.compute_basis(
                X,
                manifold_dim=manifold_dims[level],
                max_dim=max_dim,
                is_leaf=is_leaf,
                threshold=thresholds[level],
                precision=precisions[level],
                inverse=inverse
            )
            for child in node.children:
                _make_basis(child, level+1)
        _make_basis(self.root, 0)

    def make_wavelets(self, X, manifold_dims, max_dim, thresholds, precisions):
        # Recursively compute wavelet transforms for all nodes
        def _make_wavelets(node, level):
            node.make_transform(X, manifold_dims[level], max_dim, thresholds[level], precisions[level])
            for child in node.children:
                _make_wavelets(child, level+1)
        _make_wavelets(self.root, 0)

    def fgwt(self, X, manifold_dims, max_dim, thresholds, precisions):
        # Forward GMRA wavelet transform
        leafs = self.query_leaf(X)
        leafs_jk = [(leaf.node_j, leaf.node_k) for leaf in leafs]
        Qjx = [None] * X.shape[0]
        for idx, leaf in enumerate(leafs):
            x = X[idx].reshape(1, -1)
            pjx = leaf.basis @ (x.T - leaf.center)
            qjx = leaf.wav_basis @ leaf.basis.T @ pjx
            Qjx[idx] = [qjx]
            pJx = pjx
            p = path(leaf)
            for n in reversed(p[1:-1]):
                pjx = n.basis @ leaf.basis.T @ pJx + n.basis @ (leaf.center - n.center)
                qjx = n.wav_basis @ n.basis.T @ pjx
                Qjx[idx].append(qjx)
            n = p[0]
            pjx = n.basis @ leaf.basis.T @ pJx + n.basis @ (leaf.center - n.center)
            qjx = pjx
            Qjx[idx].append(qjx)
            Qjx[idx] = list(reversed(Qjx[idx]))
        return Qjx, leafs_jk

    def igwt(self, gmra_q_coeff, leaves_j_k, shape):
        # Inverse GMRA wavelet transform
        X_recon = np.zeros(shape, dtype=np.float64)
        for i in range(len(gmra_q_coeff)):
            coeffs = list(reversed(gmra_q_coeff[i]))
            leaf = self.query_leaf_by_jk(leaves_j_k[i])
            lvl_from_leaf = 0
            Qjx = leaf.wav_basis.T @ coeffs[lvl_from_leaf] + leaf.wav_consts
            leaf = leaf.parent
            lvl_from_leaf += 1
            while leaf.parent is not None:
                Qjx += (leaf.wav_basis.T @ coeffs[lvl_from_leaf] + leaf.wav_consts +
                        leaf.parent.basis.T @ leaf.parent.basis @ Qjx)
                leaf = leaf.parent
                lvl_from_leaf += 1
            Qjx += leaf.basis.T @ coeffs[lvl_from_leaf] + leaf.center
            X_recon[i:i+1,:] = Qjx.T
        return X_recon

    def query_leaf_by_jk(self, jk):
        # Use the jk_to_node dictionary for fast lookup
        return self.jk_to_node.get(jk, None)




