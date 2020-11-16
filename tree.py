from .utils import *
import numpy as np
from itertools import chain, combinations
from math import factorial
import networkx as nx

class Tree:
    """
    Class for a tree, which is primarily a wrapper for a nested structure of nodes starting at root.
    Also has a few attributes in its own right:
        name
        split_dims
        eval_dims
        transition_matrix
        transition_graph
    And methods that only make sense at the whole-tree level:
        filter()
        prune_MCCP()
        compute_transition_matrix()
        compute_transition_graph()
        dijkstra_path()
        backprop_gains()
        shap()
    """
    def __init__(self, name, root, split_dims, eval_dims):
        self.name, self.root, self.split_dims, self.eval_dims = name, root, split_dims, eval_dims
        self.leaves = self._get_leaves() # Collect the list of leaves.
        self.transition_matrix, self.transition_graph = None, None # These have to be computed explicitly.
    
    def __repr__(self):
        return f'{self.name}: split={self.split_dims}, eval={self.eval_dims}'

    def predict(self, X, dims): 
        """Propagate a set of samples through the tree and get predictions along dims."""
        # Allow dim_names to be specified instead of numbers.
        if type(dims[0]) == str: dims = [self.root.source.dim_names.index(d) for d in dims]
        # Test if just one sample has been provided.
        X = np.array(X); shp = X.shape
        if len(X.shape) == 1: X = X.reshape(1,-1)
        assert X.shape[1] == len(self.root.source.dim_names), "Must match number of dims in source."
        p = []
        # Get prediction for each sample.
        for x in X:
            leaf = self._propagate(x)
            p.append(leaf.mean[dims])
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
            var_sum = np.dot(node.var_sum, self.root.source.scale_factors)
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

    def filter(self, slice_list=[], slice_dict={}, maximise_to_slice=True, interval_list=[], interval_dict={}, max_depth=np.inf):
        """
        Filter the tree's nodes (up to a maximum depth) by either slice or interval methods, or both.
        """
        # Convert dictionary representations to lists.
        if slice_list == [] and slice_dict != {}: slice_list = dim_dict_to_list(slice_dict, self.root.source.dim_names)
        if interval_dict != {}: interval_list = dim_dict_to_list(interval_dict, self.root.source.dim_names)
        nodes = []
        def _recurse(node, depth=0):
            if node is None: return
            if depth < max_depth and node.split_dim is not None: 
                _recurse(node.left, depth+1); _recurse(node.right, depth+1)
            else:
                include = True
                if slice_list != []:
                    # Slice filtering: minimal/maximal bounding box must intersect an affine subspace. 
                    bb = node.bb_max if maximise_to_slice else node.bb_min
                    if any(bb[:,0] > [np.inf if s == None else s for s in slice_list]) or \
                       any(bb[:,1] < [-np.inf if s == None else s for s in slice_list]): include = False
                if include and interval_list != []:
                    # Interval filtering: mean must fall within an interval along each dimension.
                    if any(node.mean < [-np.inf if i == None else i[0] for i in interval_list]) or \
                       any(node.mean > [np.inf if i == None else i[1] for i in interval_list]): include = False
                if include: nodes.append(node)
        _recurse(self.root)
        return nodes

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
            leaf_idx = self.leaves.index(self._propagate(x))
            if x[time_idx] == 0: # i.e. start of an episode.
                self.transition_matrix[n, leaf_idx] += 1 # Initial.
                self.transition_matrix[leaf_idx_last, n+1] += 1 # Previous terminal.
                leaf_idx_last = leaf_idx
            elif leaf_idx != leaf_idx_last:
                self.transition_matrix[leaf_idx_last, leaf_idx] += 1 # Transition.
                leaf_idx_last = leaf_idx
        self.transition_matrix[leaf_idx_last, n+1] += 1 # Final terminal sample.
        return self.transition_matrix

    def build_transition_graph(self):
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

    def shap(self, x, dim, maximise=True): 
        """
        An implementation of TreeSHAP for computing local importances for all split_dims, based on Shapley values.
        Not as heavily-optimised as the algorithm in the original paper.
        """
        # Allow dim_name to be specified instead of number.
        if type(dim) == str: dim = self.root.source.dim_names.index(dim)
        # Store the mean value along dim, and the population, for all leaves.
        means_and_pops = {l: (l.mean[dim], l.num_samples) for l in self.leaves}
        # For each split_dim, find the set of leaves compatible with the sample's value along this dim.
        nones = [None for _ in x]
        compatible_leaves = {}
        for split_dim in self.split_dims:
            slice_list = nones.copy(); slice_list[split_dim] = x[split_dim]
            compatible_leaves[split_dim] = set(self.filter(slice_list=slice_list, maximise_to_slice=maximise))
        # Iterate through powerset of split_dims (from https://stackoverflow.com/a/1482316).
        marginals, contributions, num_split_dims = {}, {d:{} for d in self.split_dims}, len(self.split_dims)
        for dim_set in chain.from_iterable(combinations(self.split_dims, r) for r in range(num_split_dims+1)):
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
        # Finally, compute and return SHAP values.
        n_fact = factorial(num_split_dims)
        w = [factorial(i) * factorial(num_split_dims-i-1) / n_fact for i in range(0,num_split_dims)]
        return {d: sum(w[len(dim_set)] * con # weighted sum of contributions...
                for dim_set, con in c.items()) # ...from each dim_set...     
                for d, c in contributions.items()} # ...for each dim.

    def _get_leaves(self):
        leaves = []
        def _recurse(node):
            if node is None: return
            if node.split_dim is None: leaves.append(node)
            else: _recurse(node.left); _recurse(node.right)
        _recurse(self.root)
        return leaves

    def _propagate(self, x):
        def _recurse(node):
            if node.split_dim is None: return node
            else:
                if x[node.split_dim] < node.split_value: 
                    return _recurse(node.left)
                else: return _recurse(node.right)
        return _recurse(self.root)