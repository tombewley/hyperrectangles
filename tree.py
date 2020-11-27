from .utils import *
import numpy as np
from itertools import chain, combinations
from math import factorial
import networkx as nx
from tqdm import tqdm

class Tree:
    """
    Class for a tree, which is primarily a wrapper for a nested structure of nodes starting at root.
    Also has a few attributes and methods in its own right.
    """
    def __init__(self, name, root, split_dims, eval_dims):
        self.name, self.root, self.split_dims, self.eval_dims = name, root, split_dims, eval_dims
        self.leaves = self._get_leaves() # Collect the list of leaves.
        self.transition_matrix, self.transition_graph = None, None # These have to be computed explicitly.
    
    def __repr__(self):
        return f'{self.name}: split={self.split_dims}, eval={self.eval_dims}'

    def propagate(self, x, contain=False, maximise=True, max_depth=np.inf):
        """
        Propagate an input through the tree. Each element can be None (ignore),
        a scalar (treat as in prediction), or a (min, max) interval (either contain or overlap).
        Returns a set of leaves. This is effectively a generalisation of prediction.
        """
        # Convert dictionary representation to list.
        if type(x) == dict: x = dim_dict_to_list(x, self.root.source.dim_names)
        def _recurse(node, depth=0):
            if node.split_dim is None or depth >= max_depth: 
                if not maximise: 
                    # If not maximise, need to check whether we fall within bb_min.
                    for xd, lims in zip(x, node.bb_min):
                        try:
                            if xd is None or np.isnan(xd): continue
                            # For scalar.
                            if not(xd >= lims[0] and xd < lims[1]): return set() 
                        except:
                            # For (min, max) interval.
                            compare = [[i >= l for i in xd] for l in lims]
                            if (not(compare[0][0]) or compare[1][1]) if contain else (not(compare[0][1]) or compare[1][0]):
                                return set()
                return {node}
            else:
                xd, t = x[node.split_dim], node.split_threshold
                try:
                    if xd is None or np.isnan(xd):
                        # If this dim not specified, continue down both left and right.
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

    def predict(self, X, dims, maximise=False): 
        """
        Propagate a set of samples through the tree and get predictions along dims.
        """
        # Allow dim_names to be specified instead of numbers.
        if type(dims[0]) == str: dims = [self.root.source.dim_names.index(d) for d in dims]
        # Check if input has been provided in dictionary form.
        if type(X) == dict: X = [X]
        if type(X[0]) == dict: X = [dim_dict_to_list(x, self.root.source.dim_names) for x in X]
        # Check if just one sample has been provided.
        X = np.array(X); shp = X.shape
        if len(X.shape) == 1: X = X.reshape(1,-1)
        assert X.shape[1] == len(self.root.source.dim_names), "Must match number of dims in source."
        p = []
        # Get prediction for each sample.
        for x in X:
            leaves = self.propagate(x, maximise=maximise); n = len(leaves)
            if n == 0: p.append(None) # If no leaves match X.
            elif n == 1: p.append(next(iter(leaves)).mean[dims]) # If leaf uniquely determined.
            else:
                # In general, x does not uniquely determine a leaf. Compute population-weighted average.                
                p.append(weighted_average(leaves, dims))            
        return np.array(p)

    def score(self, X, dims, ord=2): 
        # Allow dim_names to be specified instead of numbers.
        if type(dims[0]) == str: dims = [self.root.source.dim_names.index(d) for d in dims]
        # Test if just one sample has been provided.
        X = np.array(X)
        if len(X.shape) == 1: X = X.reshape(1,-1)
        assert X.shape[1] == len(self.root.source.dim_names), "Must match number of dims in source."
        return np.linalg.norm(self.predict(X, dims) - X[:,dims], axis=0, ord=ord) / X.shape[0]

    def prune_MCCP(self):
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

    def compute_transition_matrix(self):
        """
        Count the transitions between all nodes and store in a matrix.
        The final two rows and cols correspond to initial/terminal.
        """
        time_idx = self.root.source.dim_names.index('time') # Must have time dimension.
        n = len(self.leaves)
        self.transition_matrix = np.zeros((n+2, n+2), dtype=int)  # Extra rows and cols for initial/terminal.
        # Iterate through the source data in temporal order.
        leaf_idx_last = n; self.transition_matrix[n, n+1] = -1 # First initial sample.
        for x in self.root.source.data:
            leaf_idx = self.leaves.index(next(iter(self.propagate(x))))
            if x[time_idx] == 0: # i.e. start of an episode.
                self.transition_matrix[n, leaf_idx] += 1 # Initial.
                self.transition_matrix[leaf_idx_last, n+1] += 1 # Previous terminal.
                leaf_idx_last = leaf_idx
            elif leaf_idx != leaf_idx_last:
                self.transition_matrix[leaf_idx_last, leaf_idx] += 1 # Transition.
                leaf_idx_last = leaf_idx
        self.transition_matrix[leaf_idx_last, n+1] += 1 # Final terminal sample.
        return self.transition_matrix

    def make_transition_graph(self):
        """
        Use transition_matrix to build a networkx graph with a node for each leaf.
        """
        # Need transition matrix first.
        if self.transition_matrix is None: self.compute_transition_matrix()
        n = len(self.leaves)
        mx = self.transition_matrix.max() # Count for single most common transition.
        G = nx.DiGraph()
        # Create nodes: one for each leaf plus initial and terminal.
        nodes = self.leaves + ['I', 'T']
        G.add_nodes_from([(l, {'idx':i if i < n else l}) for i, l in enumerate(nodes)])
        # Create edges.
        for i, node in enumerate(G.nodes): 
            count_sum = self.transition_matrix[i].sum()
            for j, count in enumerate(self.transition_matrix[i]):
                if count > 0:
                    G.add_edge(node, nodes[j], count=count, 
                                               alpha=count/mx, 
                                               cost=-np.log(count/count_sum) # Edge cost = negative log prob.
                                               )
        self.transition_graph = G
        return self.transition_graph

    def dijkstra_path(self, source, dest=None):
        """
        Use networkx's inbuild Dijktra algorithm to find the highest-probability paths from a source leaf.
        If a destination is specified, use that. Otherwise, find paths to all other leaves.
        """
        assert self.transition_graph is not None
        return nx.single_source_dijkstra(self.transition_graph, source=source, target=dest, weight="cost")

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

    def shap(self, X, wrt_dim, ignore_dim=None, maximise=True): 
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
                compatible_leaves[d] = set(self.propagate(x_s, maximise=maximise))
            # Iterate through powerset of shap_dims (from https://stackoverflow.com/a/1482316).
            marginals, contributions = {}, {d:{} for d in shap_dims}
            for dim_set in chain.from_iterable(combinations(shap_dims, r) for r in range(num_shap_dims+1)):
                if dim_set == (): mp = np.array(list(means_and_pops.values())) # All leaves.
                else:
                    matching_leaves = set.intersection(*(compatible_leaves[d] for d in dim_set)) # Leaves compatible with dim_set.
                    mp = np.array([means_and_pops[l] for l in matching_leaves]) # Means and pops as NumPy array.
                marginals[dim_set] = np.average(mp[:,0], weights=mp[:,1]) # Population-weighted average.
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

    def shap_with_ignores(self, X, wrt_dim, ignore_dims=None, maximise=True):
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