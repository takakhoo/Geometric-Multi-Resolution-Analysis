import numpy as np
# our own implementation of DyadicTree and DyadicTreeNode using Python
# DyadicTreenNode will have point_indices point to indexes in dataset, and a list of children of type DyadicTreeNode
# DyadicTree will be constructed from the CoverTree.
# root node will be at CoverTree max scale (root node).


def get_idx_sublevel(node):
    """
    Get all the index of the nodes under node
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
    def __init__(self, idxs):
        self.idxs = idxs
        self.children = []

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
        self.root = DyadicTreeNode(get_idx_sublevel(cover_tree.root))
        self.height = 1

        self.cover_tree = cover_tree
        self.build_tree(self.root, cover_tree.root)

        
    def build_tree(self, node, cover_node, level=1):

        # heigh is number of level
        # level start from 0

        if level+1 > self.height:
            self.height = level+1

        if hasattr(cover_node, 'idx'):
            child_node = DyadicTreeNode(cover_node.idx)
            node.add_child(child_node)
        else:
            for child in cover_node.children:
                child_node = DyadicTreeNode(get_idx_sublevel(child))
                node.add_child(child_node)
                self.build_tree(child_node, child, level + 1)

def tree_traverse(node, level=0, only_print_level=None):
    """
    Print the tree, starting from the root node.
    At every level, use a single dash as level indicator.
    eg: root node has '-', child node has '--', etc.
    If only_print_level is set, only print nodes at that level.

    """
    if only_print_level is None or level == only_print_level:
        print("-" * (level + 1), node.idxs)
    for child in node.children:
        tree_traverse(child, level + 1, only_print_level)
