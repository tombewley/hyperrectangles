from .node import *
from .tree import *
import numpy as np
import bisect
from tqdm import tqdm

class Source:
    """
    Master class for centrally storing data and building trees.
    """
    def __init__(self, data, dim_names):
        self.data = data
        self.dim_names = dim_names
        # Sort data along each dimension up front.
        self.all_sorted_indices = np.argsort(data, axis=0) 
        # Scale factors for variance are reciprocals of global variance.
        var = np.var(data, axis=0)
        var[var==0] = 1
        self.global_var_scale = max(var) / var
        # Empty dictionary for storing trees.
        self.trees = {}

    # Dunder/magic methods.
    def __getitem__(self, name): return self.trees[name]

    def subset(self, bb_dict=None, bb_array=None, subsample=None):
        """
        Select a subset of the data by per-dimension filtering and/or random subsampling.
        """
        sorted_indices = self.all_sorted_indices.copy()
        if bb_dict is not None:
            for split_dim, lims in enumerate(dim_dict_to_list(bb_dict, self.dim_names)):
                if lims is None: continue # If nothing specified for this lim.
                for lu, lim in enumerate(lims):
                    data = self.data[sorted_indices[:,split_dim], split_dim] # Must reselect each time.
                    split_index = bisect.bisect(data, lim)
                    left, right = split_sorted_indices(sorted_indices, split_dim, split_index)
                    if lu == 0: sorted_indices = right
                    else: sorted_indices = left
        return subsample_sorted_indices(sorted_indices, subsample)

    def tree_depth_first(self, name, split_dims, eval_dims, sorted_indices=None, max_depth=np.inf, 
                         corr=False, one_sided=False, pop_power=.5):
        """
        Grow a tree depth-first to max_depth using samples specified by sorted_indices. 
        """
        if corr: assert len(eval_dims) > 1
        split_dims, eval_dims, sorted_indices = self._preflight_check(split_dims, eval_dims, sorted_indices)
        def _recurse(node, depth):
            if node is None: return # This will be the case 50% of the time if doing one-sided.
            if depth < max_depth:
                ok, _ = node._do_greedy_split(split_dims, eval_dims, corr, one_sided, pop_power)
                if ok: _recurse(node.left, depth+1); _recurse(node.right, depth+1)
        root = Node(self, sorted_indices) 
        _recurse(root, 0)
        self.trees[name] = Tree(name, root, split_dims, eval_dims)
        return self.trees[name]

    def tree_best_first(self, name, split_dims, eval_dims, sorted_indices=None, max_num_leaves=np.inf): 
        """
        Grow a tree best-first to max_num_leaves using samples specified by sorted_indices. 
        """
        split_dims, eval_dims, sorted_indices = self._preflight_check(split_dims, eval_dims, sorted_indices)
        with tqdm(total=max_num_leaves) as pbar:
            root = Node(self, sorted_indices) 
            priority = np.dot(root.var_sum[eval_dims], self.global_var_scale[eval_dims])
            queue = [(root, priority)]
            pbar.update(1); num_leaves = 1
            while num_leaves < max_num_leaves and len(queue) > 0:
                queue.sort(key=lambda x: x[1], reverse=True)
                # Try to split the highest-priority leaf.
                node, _ = queue.pop(0) 
                ok, _ = node._do_greedy_split(split_dims, eval_dims)
                if ok:    
                    pbar.update(1); num_leaves += 1
                    # If split made, add the two new leaves to the queue.
                    queue += [(node.left,
                               np.dot(node.left.var_sum[eval_dims], self.global_var_scale[eval_dims])),
                              (node.right,
                               np.dot(node.right.var_sum[eval_dims], self.global_var_scale[eval_dims]))]
        self.trees[name] = Tree(name, root, split_dims, eval_dims)
        return self.trees[name]

    def tree_from_dict(self, name, d): 
        """
        Node attributes:
            Node(source, sorted_indices, parent_split_info) handles:
                source
                sorted_indices
                num_samples
                mean
                cov
                cov_sum
                var_sum
                bb_min
                bb_max
            Then manually add:
                split_dim = df['split_dim']
                split_index
                left = df['left']
                right = df['right']
                gains
        Tree attributes:
            Tree(name, root, split_dims, eval_dims) where:
                split_dims = unique values of df['split_dim']
                eval_dims = ?
        """
        def _recurse(node, n): 
            if n in d:
                node.split_dim = d[n]['split_dim']
                node.split_threshold = d[n]['split_threshold']
                # Most complicated aspect here is splitting the node's data at the threshold.
                data = self.data[node.sorted_indices[:,node.split_dim], node.split_dim]
                split_index = bisect.bisect(data, node.split_threshold)
                left, right = split_sorted_indices(node.sorted_indices, node.split_dim, split_index)
                node.left = Node(self, left, (node.bb_max.copy(), node.split_dim, 1, node.split_threshold))
                node.right = Node(self, right, (node.bb_max.copy(), node.split_dim, 0, node.split_threshold))
                _recurse(node.left, d[n]['left'])
                _recurse(node.right, d[n]['right'])
        root = Node(self, self.all_sorted_indices)
        _recurse(root, 1) # NOTE: Root node must have key of 1 in dict.
        split_dims, eval_dims = list(set(v['split_dim'] for v in d.values())), [] # NOTE: No eval dims?
        self.trees[name] = Tree(name, root, split_dims, eval_dims)
        return self.trees[name]

    # def question_tree(self, name, split_dims, eval_dims, sorted_indices=None, ord=1):
    #     """
    #     xxx
    #     """
    #     # assert ord == 1 and len(eval_dims) > 1
    #     split_dims, eval_dims, sorted_indices = self._preflight_check(split_dims, eval_dims, sorted_indices)
    #     root = Node(self, sorted_indices) 
    #     _, extra = root._do_greedy_split(split_dims, eval_dims, corr=True, one_sided=True)
    #     self.trees[name] = Tree(name, root, split_dims, eval_dims)

    #     qual = extra[0,:,:,0,1]

    #     return self.trees[name], qual

    def _preflight_check(self, split_dims, eval_dims, sorted_indices):
        # Allow dim_names to be specified instead of numbers.
        if type(split_dims[0]) == str: split_dims = [self.dim_names.index(s) for s in split_dims]
        if type(eval_dims[0]) == str: eval_dims = [self.dim_names.index(e) for e in eval_dims] 
        # If indices not specified, use all.
        if sorted_indices is None: sorted_indices = self.all_sorted_indices
        return np.array(split_dims), np.array(eval_dims), sorted_indices