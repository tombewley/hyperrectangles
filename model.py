from .utils import *
import numpy as np
import networkx as nx
from itertools import chain, combinations
from math import factorial
from tqdm import tqdm

class Model:
    """
    Class for a model, which is a wrapper for a flat set of leaves. 
    Tree inherits from here, overwrites some methods and adds new ones.
    """
    def __init__(self, name, leaves):
        self.name = name
        if leaves: self.leaves, self.source = leaves, leaves[0].source
        # These attributes are computed on request.
        self.transition_matrix, self.transition_graph = None, None 
    
    def propagate(self, x, contain=False, mode='min', max_depth=np.inf):
        """
        Iterate through all leaves and check whether an input x is inside.
        """
        return {leaf for leaf in self.leaves if is_x_in_node(leaf, x, contain, mode)}

    def predict(self, X, dims, maximise=False): 
        """
        Propagate a set of samples through the model and get predictions along dims.
        """
        # Allow dim_names to be specified instead of numbers.
        if type(dims[0]) == str: dims = [self.source.dim_names.index(d) for d in dims]
        # Check if input has been provided in dictionary form.
        if type(X) == dict: X = [X]
        if type(X[0]) == dict: X = [dim_dict_to_list(x, self.source.dim_names) for x in X]
        # Check if just one sample has been provided.
        X = np.array(X); shp = X.shape
        if len(X.shape) == 1: X = X.reshape(1,-1)
        assert X.shape[1] == len(self.source.dim_names), "Must match number of dims in source."
        p = []
        # Get prediction for each sample.
        for x in X:
            leaves = self.propagate(x, mode=('max' if maximise else 'min')); n = len(leaves)
            if n == 0: p.append([None for _ in dims]) # If no leaves match X.
            elif n == 1: p.append(next(iter(leaves)).mean[dims]) # If leaf uniquely determined.
            else:
                # In general, x does not uniquely determine a leaf. Compute population-weighted average.                
                p.append(weighted_average(leaves, dims))            
        return np.array(p)

    def score(self, X, dims, ord=2): 
        # Allow dim_names to be specified instead of numbers.
        if type(dims[0]) == str: dims = [self.source.dim_names.index(d) for d in dims]
        # Test if just one sample has been provided.
        X = np.array(X)
        if len(X.shape) == 1: X = X.reshape(1,-1)
        assert X.shape[1] == len(self.source.dim_names), "Must match number of dims in source."
        return np.linalg.norm(self.predict(X, dims) - X[:,dims], axis=0, ord=ord) / X.shape[0]

    def counterfactual(self, x, foil, delta_dims, fixed_dims=[],
                       access_mode='min', sort_by='L0_L2', return_all=False):
        """
        Return a list of minimal counterfactuals from x given foil, sorted by the provided method.
        """
        foil = dim_dict_to_list(foil, self.source.dim_names)
        if type(delta_dims[0]) == str: delta_dims = [self.source.dim_names.index(d) for d in delta_dims]
        if fixed_dims and type(fixed_dims[0]) == str: fixed_dims = [self.source.dim_names.index(d) for d in fixed_dims]
        # Marginalise out all non-fixed dims in x.
        x_marg = x.copy()
        x_marg[[d for d in range(len(x)) if d not in fixed_dims]] = None
        # Accessible leaves are those that intersect the marginalised x.
        leaves_accessible = self.propagate(x_marg, mode=access_mode)
        # Foil leaves are those that intersect the foil condition (mean mode).
        leaves_foil = self.propagate(foil, mode='mean') 
        # We are intersected in leaves that are both accessible and foils.
        options = []
        scale = np.sqrt(self.source.global_var_scale[delta_dims])
        for leaf in leaves_accessible & leaves_foil:
            # Find the closest point in each.
            x_closest = closest_point_in_bb(x, leaf.bb_min if access_mode=='min' else leaf.bb_max)
            # Compute the L0 and L2 norms. NOTE: normalise each dim by global standard deviation!
            delta = (x_closest - x)[delta_dims] * scale
            l0, l2 = np.linalg.norm(delta, ord=0), np.linalg.norm(delta, ord=2)
            options.append((leaf, x_closest, l0, l2))
        # Sort foil options by L0, then by L2.
        if sort_by == 'L0_L2': options.sort(key=lambda x: (x[2], x[3]))            
        # Sort foil options by L2.  
        elif sort_by == 'L2': options.sort(key=lambda x: x[3])   
        return options if return_all else options[0]

    def make_transition_matrix(self):
        """
        Count the transitions between all nodes and store in a matrix.
        The final two rows and cols correspond to initial/terminal.
        """
        time_idx = self.source.dim_names.index('time') # Must have time dimension.
        n = len(self.leaves)
        self.transition_matrix = np.zeros((n+2, n+2), dtype=int)  # Extra rows and cols for initial/terminal.
        # Iterate through the source data in temporal order.
        leaf_idx_last = n; self.transition_matrix[n, n+1] = -1 # First initial sample.
        for x in self.source.data:
            leaves = self.propagate(x, mode='max') # NOTE: Do we really need to propagate here?
            assert len(leaves) == 1, "Can only compute transitions if leaves are disjoint and exhaustive."
            leaf_idx = self.leaves.index(next(iter(leaves)))
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
        if self.transition_matrix is None: self.make_transition_matrix()
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

    def shap(self, X, shap_dims, wrt_dim, ignore_dim=None, maximise=False): 
        """
        An implementation of TreeSHAP for computing local importances for shap_dims, based on Shapley values.
        NOTE: Not as heavily-optimised as the algorithm in the original paper.
        """
        # Allow dim_names to be specified instead of numbers.
        if type(shap_dims[0]) == str: shap_dims = [self.source.dim_names.index(d) for d in shap_dims]
        if type(wrt_dim) == str: wrt_dim = self.source.dim_names.index(wrt_dim)
        shap_dims = set(shap_dims)
        assert wrt_dim not in shap_dims, "Can't include wrt_dim in the set of shap_dims!"
        if ignore_dim is not None:
            if type(ignore_dim) == str: ignore_dim = self.source.dim_names.index(ignore_dim)
            shap_dims -= {ignore_dim}
        # Store the mean value along wrt_dim, and the population, for all leaves.
        means_and_pops = {l: (l.mean[wrt_dim], l.num_samples) for l in self.leaves}
        # Pre-store some reused values.
        nones = [None for _ in self.source.dim_names]
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

    def shap_with_ignores(self, X, shap_dims, wrt_dim, ignore_dims=None, maximise=False):
        """This function allows us to calculate pairwise SHAP interaction values."""
        if ignore_dims is None: ignore_dims = shap_dims
        shaps = {}
        for ignore_dim in set(ignore_dims) | {None}:
            print(f'Ignoring {ignore_dim}...')
            shaps[ignore_dim] = self.shap(X, shap_dims, wrt_dim, ignore_dim=ignore_dim, maximise=maximise)
        return shaps