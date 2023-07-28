from .model import Model
from .node import Node
from .utils import *
import numpy as np

class Tree(Model):
    """
    Class for a tree, which inherits from model and introduces a few tree-specific methods.
    """
    def __init__(self, name, root, split_dims=None, eval_dims=None, split_finder=variance_based_split_finder):
        Model.__init__(self, name, leaves=None) # Don't explicitly pass leaves because they're under root.
        self.root, self.space, self.split_dims, self.eval_dims = root, root.space, split_dims, eval_dims
        self.leaves = self._get_nodes(leaves_only=True) # Collect the list of leaves.
        self.split_finder = split_finder
        if split_dims is not None: self._compute_split_queue()

    # Dunder/magic methods.
    def __repr__(self): return f"{self.name}: tree model with {len(self)} leaves (split: {self.split_dims}, eval: {self.eval_dims})"
    def __sub__(self, other): return self.diff(other)

    @property
    def siblings(self):
        """
        Return the indices of all sibling pairs in self.leaves.
        """
        return [(x, x+1) for x in range(len(self.leaves)-1) if self.leaves[x].parent is self.leaves[x+1].parent]

    def _compute_split_queue(self):
        """
        Compute split queue for best-first growth from scratch, and empty split cache.
        """
        self.split_queue = [(leaf, np.dot(leaf.var_sum[self.eval_dims], self.space.global_var_scale[self.eval_dims])) for leaf in self.leaves]
        self.split_queue.sort(key=lambda x: x[1], reverse=True)
        self.split_cache, self.split_skipped = [], set()

    def populate(self, sorted_indices="all", keep_hr_min=False):
        """
        Populate all nodes in the tree with data from a sorted_indices array.
        Then recompute the split queue.
        """
        if sorted_indices == "all": sorted_indices = self.space.all_sorted_indices
        def _recurse(node, si):
            node.populate(si, keep_hr_min)
            if node.split_dim is None: return
            if sorted_indices is None: left, right = None, None
            else:
                split_index = bisect.bisect(self.space.data[si[:,node.split_dim], node.split_dim], node.split_threshold)
                left, right = split_sorted_indices(si, node.split_dim, split_index)
            _recurse(node.left, left); _recurse(node.right, right)
        _recurse(self.root, sorted_indices)
        self._compute_split_queue()
        return self

    def propagate(self, x, mode, contain=False, max_depth=np.inf, path=False):
        """
        Overwrites Model.propagate using a more efficient tree-specific method.
        """
        if mode == "fuzzy": raise NotImplementedError()
        if path and mode != "max": raise NotImplementedError("Can only return path in maximise mode.")
        x = self.space.listify(x)
        assert len(x) == len(self.space), f"Input is {len(x)} dimensional; space is {len(self.space)} dimensional."
        def _recurse(node, depth=0):
            if node is None: return set()
            if node.split_dim is None or depth >= max_depth: 
                # If have reached a leaf and mode is min or mean, still need to check compatibility.
                if mode in ("min", "mean") and not node.membership(x, mode, contain): return set()
                return {node}
            else:
                xd, t = x[node.split_dim], node.split_threshold
                try:
                    if xd is None or np.isnan(xd):
                        # If this dim unspecified (None), continue down both left and right.
                        return _recurse(node.left, depth+1) | _recurse(node.right, depth+1)
                    # If this dim has a scalar, compare to the threshold.
                    if xd >= t: 
                        return _recurse(node.right, depth+1) | ({node} if path else set())
                    else: return _recurse(node.left, depth+1) | ({node} if path else set())
                except:
                    # If this dim has (min, max) interval, compare each to the threshold.
                    compare = [i >= t for i in xd]
                    if (not(compare[1]) if contain else not(compare[0])):
                        left = _recurse(node.left, depth+1)
                    else: left = set()
                    if (compare[0] if contain else compare[1]):
                        right = _recurse(node.right, depth+1)
                    else: right = set()
                    return left | right | ({node} if path else {})
        return _recurse(self.root)

    def get_leaf_nums(self, X, one_hot=False):
        """
        Accelerated method for propagating a multidimensional array of samples
        through the tree and getting the unique leaf num for each.
        Optionally use a one-hot encoding.
        """
        # First flatten along all but the final dimension to ease indexing.
        shape = X.shape; X_flat = X.reshape(-1, shape[-1])
        if one_hot:
            leaf_nums = np.full((*shape[:-1], len(self)), -1)
            oh = np.identity(len(self), dtype=int)
        else: leaf_nums = np.full(shape[:-1], -1)
        def _recurse(node, indices):
            if len(indices) == 0: return
            if node.split_dim is None:
                # At a leaf, store the leaf num for all remaining indices.
                x = self.leaves.index(node)
                leaf_nums[np.unravel_index(indices, shape[:-1])] = oh[x] if one_hot else x
            else:
                # At an internal node, split the indices based on the split threshold.
                left = X_flat[indices, node.split_dim] < node.split_threshold
                _recurse(node.left, indices[left]); _recurse(node.right, indices[~left])
        _recurse(self.root, np.arange(len(X_flat)))
        return leaf_nums

    def _queue_to_cache(self, min_samples_leaf, entropy=0., store_all_qual=False):
        """
        Find the greedy split for the first leaf in the split queue and add to the split cache.
        """
        node, _ = self.split_queue.pop(0) 
        self.split_cache.append((node, node._find_greedy_split(self.split_finder, self.split_dims, self.eval_dims, min_samples_leaf, entropy, store_all_qual)))
        self.split_cache.sort(key=lambda x: x[1][2], reverse=True)
        assert set(self.leaves) == set([n for n, _ in self.split_queue]) | set([n for n, _ in self.split_cache]) | self.split_skipped

    def split_next_best(self, min_samples_leaf, num_from_queue=np.inf, entropy=0., store_all_qual=False):
        """
        Split the next leaf.
        The num_from_queue argument facilitates a tradeoff between heuristically splitting based on var_sum (set to 1)
        and exhaustively trying every leaf (set to inf).
        """
        for _ in range(min(num_from_queue, len(self.split_queue))):
            self._queue_to_cache(min_samples_leaf, entropy, store_all_qual) # Transfer the first leaf in the split queue to the cache.
        node, (split_dim, split_index, qual, gains) = self.split_cache.pop(0)
        if qual > 0: 
            node._do_split(split_dim, split_index=split_index, gains=gains)
            # If split made, store the two new leaves and add them to the queue.
            self.leaves = self._get_nodes(leaves_only=True) # NOTE: Doing it this way preserves a consistent ordering scheme.            
            self.split_queue += [(node.left,  np.dot(node.left.var_sum[self.eval_dims], self.space.global_var_scale[self.eval_dims])),
                                 (node.right, np.dot(node.right.var_sum[self.eval_dims], self.space.global_var_scale[self.eval_dims]))]
            self.split_queue.sort(key=lambda x: x[1], reverse=True) # Sort ready for next time.
            self.split_skipped = set()
            return node
        self.split_skipped.add(node)

    def find_dca(self, nodes):
        """
        Find the deepest common ancestor node of a set of nodes.
        """
        nodes = set(nodes)
        def _recurse(node=self.root):
            subtree = {node}
            found = nodes == subtree
            if not found and node.split_dim is not None:
                found_left, left = _recurse(node.left)
                if found_left: return found_left, left # Already found in left subtree.
                found_right, right = _recurse(node.right)
                if found_right: return found_right, right # Already found in right subtree.
                subtree = subtree | left | right
            if not (nodes - subtree): found = True # Found if all nodes are in subtree.
            return found, (node if found else subtree)
        found, dca = _recurse()
        return dca if found else None

    def dca_subtree(self, name, nodes):
        """
        Return a subtree rooted at the dca for a set of nodes, pruned so that subtree.leaves = nodes.
        """
        # First find the dca.
        dca = self.find_dca(nodes)
        if dca is None: return None
        # Next iterate through the subtree rooted at dca and iteratively replace nodes with one child,
        # using that child itself. This is not quite the same as pruning.
        subtree_split_dims = set()
        def _recurse_minimise(node):
            replacement = node if node in nodes_copy else None
            if node.split_dim is not None:
                replacement_left = _recurse_minimise(node.left)
                replacement_right = _recurse_minimise(node.right)
                if replacement_left != node.left: 
                    # Replace the left child, either with None or one of its children.
                    hr_max_left = node.left.hr_max
                    node.left = replacement_left 
                    # Keep existing hyperrectangle.
                    if replacement_left is not None: node.left.hr_max = hr_max_left
                if replacement_right != node.right: 
                    # Replace the right child, either with None or one of its children.
                    hr_max_right = node.right.hr_max
                    node.right = replacement_right 
                    # Keep existing hyperrectangle.
                    if replacement_right is not None: node.right.hr_max = hr_max_right
                if replacement is None:
                    # Determine how to replace this node.
                    if node.left is None:
                        if node.right is not None: replacement = node.right 
                    elif node.right is None: replacement = node.left
                    else: replacement = node      
                # Store split_dims for subtree; generally a subset of those used in this tree.
                if replacement is not None: subtree_split_dims.add(replacement.split_dim)
            return replacement
        # Using deepcopy ensures that this tree is not affected when we mess with the subtree.
        from copy import deepcopy
        dca_copy, nodes_copy, eval_dims_copy = deepcopy((dca, nodes, self.eval_dims))
        subtree_root = _recurse_minimise(dca_copy)
        # Remove parent from subtree root, and split information from subtree leaves.
        subtree_root.parent = None
        for leaf in nodes_copy: leaf.unsplit()
        subtree_split_dims = sorted(list(subtree_split_dims - {None}))
        return Tree(name, subtree_root, subtree_split_dims, eval_dims_copy)

    def diff(self, other, resolution=0):
        """
        Return a "differential tree".
        """
        assert self.space.dim_names == other.space.dim_names, "Trees must be in equivalent spaces."

        root = Node(self.space.empty_clone())

        active_nodes = {root}
        self_node = self.root
        other_node = other.root

        print(self_node.split_dim, self_node.split_threshold)
        print(other_node.split_dim, other_node.split_threshold)

        print(active_nodes)
        active_children = set()
        for node in active_nodes: 
            node._do_split(self_node.split_dim, self_node.split_threshold)
            if self_node.split_dim == other_node.split_dim:
                if abs(self_node.split_threshold - other_node.split_threshold) <= resolution:
                    # Case A: Add no children; branching factor = 2
                    pass
                else:
                    # Case B: Add one child; branching factor = 3
                    pass
            else:
                # Case C: Add both children; branching factor = 4
                pass
            
        print(active_children)
        for child in active_children: 
            child._do_split(other_node.split_dim, other_node.split_threshold)
    
        split_dims_union = []
        return Tree(f"{self.name}_sub_{other.name}", root, split_dims_union, [])

    def prune_mccp(self):
        """
        Perform one step of minimal cost complexity pruning.
        See http://mlwiki.org/index.php/Cost-Complexity_Pruning for details.
        Here, cost = reduction in var_sum / (num leaves in subtree - 1).
        NOTE: A full pruning sequence is slightly inefficient because have to
        recompute costs on each iteration, but there are advantages to modularity.
        """
        # Subfunction for calculating costs is similar to the _recurse() function inside backprop_gains(),
        # except it takes the weighted sum of var_sum rather than per-feature, and realised only.
        def _recurse(node):
            var_sum = np.dot(node.var_sum[self.eval_dims], self.space.global_var_scale[self.eval_dims])
            if node.split_dim is None: return [var_sum], 1
            (left, num_left), (right, num_right) = _recurse(node.left), _recurse(node.right)
            var_sum_leaves, num_leaves = left + right, num_left + num_right
            costs.append((node, (var_sum - sum(var_sum_leaves)) / (num_leaves - 1), sum(var_sum_leaves), num_leaves)) 
            return var_sum_leaves, num_leaves
        costs = []
        _recurse(self.root)
        costs.sort(key=lambda x: x[1])
        # Prune the subtree below the lowest-cost node.
        node = costs[0][0]
        return self._get_nodes().index(node), self.prune_to(node)

    def prune_to(self, node):
        """
        Prune to a specified node and return pruned leaf nums for reference.
        """
        pruned_leaf_nums = {self.leaves.index(l) for l in self._get_nodes(root=node, leaves_only=True)}
        node.split_dim, node.split_threshold, node.left, node.right, node.gains = None, None, None, None, {}
        # Update the list of leaves and split queue.
        self.leaves = self._get_nodes(leaves_only=True) 
        self._compute_split_queue()
        return pruned_leaf_nums

    def feature_importance(self):
        return self.root.feature_importance(self.split_dims, self.eval_dims)

    def backprop_gains(self):
        """
        Propagate gains for each splitting feature back through the tree, 
        enabling a hierarchical analysis of feature importance.
        """
        assert self.root.gains is not None
        def _recurse(node):
            if node.split_dim is None: return {'realised':0., 'potential': 0.}, 1
            # Realised gains are those for the actually-chosen split only.
            node.gains['realised'] = np.zeros_like(node.gains['immediate'])
            d = np.argwhere(self.split_dims == node.split_dim)[0,0]
            node.gains['realised'][d] = node.gains['immediate'][d]
            # Add realised and potential gains from children.
            (left, num_left), (right, num_right) = _recurse(node.left), _recurse(node.right)
            node.gains['realised'] += left['realised'] + right['realised']
            node.gains['potential'] = node.gains['immediate'] + left['potential'] + right['potential']
            # Compute relative gains by normalising by var_sum at this node.
            here = node.var_sum[self.eval_dims]
            here[here == 0] = 1 # Prevents div/0 error.
            node.gains['realised_relative'] = node.gains['realised'] / here
            node.gains['potential_relative'] = node.gains['potential'] / here
            # Compute alpha values (as used in cost complexity pruning) by normalising by number of leaves in the subtree.
            node.subtree_size = num_left + num_right
            node.gains['realised_alpha'] = node.gains['realised'] / (node.subtree_size - 1)
            node.gains['potential_alpha'] = node.gains['potential'] / (node.subtree_size - 1)
            return node.gains, node.subtree_size
        _recurse(self.root)

    def clone(self):     
        """
        Clone this tree, retaining only the reference to the space.
        """
        from copy import deepcopy
        clone = deepcopy(self)
        clone.space = self.space
        def _recurse(node):
            node.space = self.space
            if node.split_dim is None: return
            _recurse(node.left); _recurse(node.right)
        _recurse(clone.root)
        return clone

    def _get_nodes(self, root=None, leaves_only=False):
        nodes = []
        def _recurse(node):
            if node is None: return
            if node.split_dim is not None: 
                if not leaves_only: nodes.append(node)
                _recurse(node.left); _recurse(node.right)
            else: nodes.append(node)
        _recurse(self.root if root is None else root)
        return nodes
