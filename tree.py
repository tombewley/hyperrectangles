from .model import Model
from .utils import *
import numpy as np
from itertools import chain, combinations
from math import factorial
from tqdm import tqdm

class Tree(Model):
    """
    Class for a tree, which inherits from model and introduces a few tree-specific methods.
    """
    def __init__(self, name, root, split_dims, eval_dims):
        Model.__init__(self, name, leaves=None) # Don't explicitly pass leaves. 
        self.root, self.split_dims, self.eval_dims = root, split_dims, eval_dims
        self.leaves = self._get_leaves() # Collect the list of leaves.

    def propagate(self, x, contain=False, mode='min', max_depth=np.inf):
        """
        Overwrites Model.propagate using a more efficient tree-specific method.
        """
        # Convert dictionary representation to list.
        if type(x) == dict: x = dim_dict_to_list(x, self.root.source.dim_names)
        def _recurse(node, depth=0):
            if node is None: return set()
            if node.split_dim is None or depth >= max_depth: 
                # If have reached a leaf and mode is min or mean, still need to check compatibility.
                if mode in ('min', 'mean') and not is_x_in_node(node, x, contain, mode): return set()
                return {node}
            else:
                xd, t = x[node.split_dim], node.split_threshold
                try:
                    if xd is None or np.isnan(xd):
                        # If this dim unspecified (None), continue down both left and right.
                        return _recurse(node.left, depth+1) | _recurse(node.right, depth+1)
                    # If this dim has a scalar, compare to the threshold.
                    if xd >= t: 
                        return _recurse(node.right, depth+1)
                    else: return _recurse(node.left, depth+1)
                except:
                    # If this dim has (min, max) interval, compare each to the threshold.
                    compare = [i >= t for i in xd]
                    if (not(compare[1]) if contain else not(compare[0])):
                        left = _recurse(node.left, depth+1)
                    else: left = set()
                    if (compare[0] if contain else compare[1]):
                        right = _recurse(node.right, depth+1)
                    else: right = set()
                    return left | right
        return _recurse(self.root)

    def dca_subtree(self, name, nodes): 
        """ 
        Find the deepest common ancestor node of a set of nodes.
        Return a subtree rooted at this node, pruned so that subtree.leaves = nodes.
        """
        # Use deepcopy to ensure that this tree itself is not affected
        # when we mess with the extracted subtree.
        from copy import deepcopy
        self_prev = deepcopy(self)
        # First find the dca and create a subtree rooted here.
        def _recurse_find_dca(node):
            subtree = {node}
            found = nodes == subtree
            if not found and node.split_dim:
                found_left, left = _recurse_find_dca(node.left)
                if found_left: return found_left, left # Already found in left subtree.
                found_right, right = _recurse_find_dca(node.right)
                if found_right: return found_right, right # Already found in right subtree.
                subtree = subtree | left | right
            if not (nodes - subtree): found = True # Found if all nodes are in subtree.
            return found, (node if found else subtree)
        found, dca = _recurse_find_dca(self.root)
        if not found: return False 
        subtree = Tree(name, dca, self.split_dims, self.eval_dims)

        print('TREE BEFORE', len(self.leaves))
        print('SUBTREE BEFORE', len(subtree.leaves))

        # Next iterate through the subtree and iteratively replace nodes with one child,
        # using that child itself. This is not quite the same as pruning.
        def _recurse_minimise(node, path=''):

            replacement = node if node in nodes else None

            if node.split_dim:
                replacement_left = _recurse_minimise(node.left, path+'0')
                replacement_right = _recurse_minimise(node.right, path+'1')
                if replacement_left != node.left: 
                    # Replace the left child, either with None or one of its children.
                    bb_max_left = node.left.bb_max
                    node.left = replacement_left 
                    # Keep existing bounding box.
                    if replacement_left is not None: node.left.bb_max = bb_max_left

                if replacement_right != node.right: 
                    # Replace the right child, either with None or one of its children.
                    bb_max_right = node.right.bb_max
                    node.right = replacement_right 
                    # Keep existing bounding box.
                    if replacement_right is not None: node.right.bb_max = bb_max_right

                if replacement is None:
                    # Determine how to replace this node.
                    if node.left is None:
                        if node.right is not None: replacement = node.right
                    elif node.right is None: replacement = node.left
                    else: replacement = node

                # print('PATH', path)
                # print(node)
                # print(replacement_left, replacement_right)
                # # print(node.left, node.right)
                # print(replacement)
                # print()
      
            return replacement

        subtree.root = _recurse_minimise(subtree.root)
        subtree.leaves = subtree._get_leaves() # Recompute leaves.
        print('SUBTREE AFTER', len(subtree.leaves))
        self = self_prev # Restore deepcopy.
        print('TREE AFTER', len(self._get_leaves()))
        return subtree

    def prune_mccp(self):
        """
        Perform one step of minimal cost complexity pruning.
        See http://mlwiki.org/index.php/Cost-Complexity_Pruning for details.
        Here, cost = reduction in var_sum / (num leaves in subtree - 1).
        NOTE: A full pruning sequence is slightly inefficient
        because have to recompute costs on each iteration, 
        but there are advantages to modularity.
        """
        # Subfunction for calculating costs is similar to the _recurse() function inside backprop_gains(),
        # except it takes the weighted sum of var_sum rather than per-feature, and realised only.
        def _recurse(node):
            var_sum = np.dot(node.var_sum, self.root.source.global_var_scale)
            if node.split_dim is None: return [var_sum], 1
            (left, num_left), (right, num_right) = _recurse(node.left), _recurse(node.right)
            var_sum_leaves, num_leaves = left + right, num_left + num_right
            costs.append((node, (var_sum - sum(var_sum_leaves)) / (num_leaves - 1), sum(var_sum_leaves), num_leaves)) # ))
            return var_sum_leaves, num_leaves
        costs = []
        _recurse(self.root)
        # Remove the subtree below the lowest-cost node.
        node = sorted(costs, key=lambda x: x[1])[0][0]
        node.split_dim, node.left, node.right, node.gains = None, None, None, {}
        self.leaves = self._get_leaves() # Update the list of leaves.

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

    def shap(self, X, wrt_dim, ignore_dim=None, maximise=False): 
        """
        An implementation of TreeSHAP for computing local importances for split_dims, based on Shapley values.
        NOTE: Not as heavily-optimised as the algorithm in the original paper.
        """
        # Allow dim_name to be specified instead of number.
        if type(wrt_dim) == str: wrt_dim = self.root.source.dim_names.index(wrt_dim)
        shap_dims = set(self.split_dims)
        if ignore_dim is not None:
            if type(ignore_dim) == str: ignore_dim = self.root.source.dim_names.index(ignore_dim)
            shap_dims -= {ignore_dim}
        # Store the mean value along wrt_dim, and the population, for all leaves.
        means_and_pops = {l: (l.mean[wrt_dim], l.num_samples) for l in self.leaves}
        # Pre-store some reused values.
        nones = [None for _ in self.root.source.dim_names]
        num_shap_dims = len(shap_dims)
        w = [factorial(i) * factorial(num_shap_dims-i-1) / factorial(num_shap_dims) 
             for i in range(0,num_shap_dims)]
        shaps = []
        for x in tqdm(X):
            # For each split_dim, find the set of leaves compatible with the sample's value along this dim.
            compatible_leaves = {}
            for d in shap_dims:
                x_s = nones.copy(); x_s[d] = x[d]
                compatible_leaves[d] = set(self.propagate(x_s, mode=('max' if maximise else 'min')))
            # Iterate through powerset of shap_dims (from https://stackoverflow.com/a/1482316).
            marginals, contributions = {}, {d:{} for d in shap_dims}
            for dim_set in chain.from_iterable(combinations(shap_dims, r) for r in range(num_shap_dims+1)):
                if dim_set == (): mp = np.array(list(means_and_pops.values())) # All leaves.
                else:
                    matching_leaves = set.intersection(*(compatible_leaves[d] for d in dim_set)) # Leaves compatible with dim_set.
                    mp = np.array([means_and_pops[l] for l in matching_leaves]) # Means and pops as NumPy array.
                try: marginals[dim_set] = np.average(mp[:,0], weights=mp[:,1]) # Population-weighted average.
                except:
                    print(x)
                    print(dim_set)
                    print([len(compatible_leaves[d]) for d in dim_set])
                    print(self.propagate(x, mode=('max' if maximise else 'min')))
                    print(mp)


                    raise Exception()


                # For each dim in the dim_set, compute the effect of adding it.
                if len(dim_set) > 0:
                    for i, d in enumerate(dim_set):
                        dim_set_without = dim_set[:i]+dim_set[i+1:]
                        contributions[d][dim_set_without] = marginals[dim_set] - marginals[dim_set_without]
            # Finally, compute SHAP values.
            shaps.append({d: sum(w[len(dim_set)] * con # weighted sum of contributions...
                          for dim_set, con in c.items()) # ...from each dim_set...     
                          for d, c in contributions.items()}) # ...for each dim.
        return shaps

    def shap_with_ignores(self, X, wrt_dim, ignore_dims=None, maximise=False):
        """This function allows us to calculate pairwise SHAP interaction values."""
        if ignore_dims is None: ignore_dims = self.split_dims
        shaps = {}
        for ignore_dim in set(ignore_dims) | {None}:
            print(f'Ignoring {ignore_dim}...')
            shaps[ignore_dim] = self.shap(X, wrt_dim, ignore_dim=ignore_dim, maximise=maximise)
        return shaps

    def _get_leaves(self):
        leaves = []
        def _recurse(node):
            if node is None: return
            if node.split_dim is None: leaves.append(node)
            else: _recurse(node.left); _recurse(node.right)
        _recurse(self.root)
        return leaves