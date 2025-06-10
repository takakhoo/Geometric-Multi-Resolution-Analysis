import numpy as np
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

        self.node_k = None  # this will be used to store the node in the CoverTree
        self.node_j = None  # this will be used to store the node in the CoverTree

    def add_child(self, child_node):
        self.children.append(child_node)

    def __getitem__(self, index):
        return self.children[index]
    
    def __len__(self):
        return len(self.children)
    
class DyadicTree:
    def __init__(self, cover_tree):
        if cover_tree.n == 0:
            raise ValueError("Cover tree is empty")
        self.root = DyadicTreeNode(get_idx_sublevel(cover_tree.root), parent=None)
        self.height = 1

        # underlying cover tree
        self.cover_tree = cover_tree
        self.idx_to_leaf_node = {}

        self.build_tree(self.root, cover_tree.root)

        # a idx to node mapping, every idx is essentially a point
        # we want to have idx -> node

        
    def build_tree(self, node, cover_node, level=1):
        """
        Recursively build the DyadicTree from the CoverTree. 
        Remember to update idx_to_leaf_node mapping.
        """


        if level+1 > self.height:
            self.height = level+1

        if hasattr(cover_node, 'idx'):
            # the leaf node will have idx
            child_node = DyadicTreeNode(cover_node.idx, parent=node)

            self.idx_to_leaf_node[cover_node.idx[0]] = child_node

            node.add_child(child_node)
        else:
            for child in cover_node.children:
                child_node = DyadicTreeNode(get_idx_sublevel(child), parent=node)
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

        
        

        