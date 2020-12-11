from .node import *
from .tree import *
import numpy as np
import bisect
from tqdm import tqdm

class Source:
    """
    Master class for centrally storing data and building models.
    """
    def __init__(self, data, dim_names):
        self.data = data
        self.dim_names = dim_names
        # Sort data along each dimension up front.
        self.all_sorted_indices = np.argsort(data, axis=0) 
        # Scale factors for variance are reciprocals of global variance.
        var = np.var(data, axis=0)
        var[var==0] = 1 # Prevent div/0 error.
        self.global_var_scale = 1 / var
        # Empty dictionary for storing models.
        self.models = {}

    # Dunder/magic methods.
    def __getitem__(self, name): return self.models[name]

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
        self.models[name] = Tree(name, root, split_dims, eval_dims)
        return self.models[name]

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
        self.models[name] = Tree(name, root, split_dims, eval_dims)
        return self.models[name]

    def tree_from_dict(self, name, d): 
        """
        Create a tree from a dictionary object.
        """
        def _recurse(node, n): 
            if n in d:
                if not node._do_manual_split(d[n]['split_dim'], d[n]['split_threshold']):
                    raise ValueError(f"Invalid split threshold for node {n}: \"{d[n]}\".")
                _recurse(node.left, d[n]['left'])
                _recurse(node.right, d[n]['right'])
        root = Node(self, self.all_sorted_indices)
        _recurse(root, 1) # Root node must have key of 1 in dict.
        split_dims, eval_dims = list(set(v['split_dim'] for v in d.values())), [] # NOTE: No eval dims.
        self.models[name] = Tree(name, root, split_dims, eval_dims)
        return self.models[name]

    def tree_from_func(self, name, func):
        """
        Create a tree from a well-formed nested if-then function in Python.
        Tests must use the < operator; split_dims can either be identified with indices, e.g. x[0],
        or with a valid entry in self.dim_names.
        """
        from dill.source import getsource
        lines = [l.strip() for l in getsource(func).split("\n")[:-1]]
        assert lines[0][:3] == "def"
        def _recurse(node, n):
            if lines[n][0] == "#": return _recurse(node, n + 1) 
            elif lines[n][:2] == "if":
                d, o, t = lines[n][3:-1].split(" ")
                assert o in ("<", ">=")
                try: split_dim = int(d.split("[")[1][:-1]) # If index specified.
                except: split_dim = self.dim_names.index(d) # If dim_name specified.
                split_dims.add(split_dim)
                if not node._do_manual_split(split_dim, float(t)):
                    raise ValueError(f"Invalid split threshold at line {n}: \"{lines[n]}\".")
                n = _recurse(node.left if o == "<" else node.right, n + 1)
                assert lines[n] == "else:"
                n = _recurse(node.right if o == "<" else node.left, n + 1)
            elif lines[n][:6] == "return": n += 1 # NOTE: Not doing anything with returns.
            else: raise ValueError(f"Parse error at line {n}: \"{lines[n]}\".")
            return n
        split_dims, eval_dims = set(), [] # NOTE: No eval dims.
        root = Node(self, self.all_sorted_indices)
        _recurse(root, 1)
        self.models[name] = Tree(name, root, sorted(split_dims), eval_dims)
        return self.models[name]

    # def question_tree(self, name, split_dims, eval_dims, sorted_indices=None, ord=1):
    #     """
    #     xxx
    #     """
    #     # assert ord == 1 and len(eval_dims) > 1
    #     split_dims, eval_dims, sorted_indices = self._preflight_check(split_dims, eval_dims, sorted_indices)
    #     root = Node(self, sorted_indices) 
    #     _, extra = root._do_greedy_split(split_dims, eval_dims, corr=True, one_sided=True)
    #     self.models[name] = Tree(name, root, split_dims, eval_dims)

    #     qual = extra[0,:,:,0,1]

    #     return self.models[name], qual

    def _preflight_check(self, split_dims, eval_dims, sorted_indices):
        # Allow dim_names to be specified instead of numbers.
        if type(split_dims[0]) == str: split_dims = [self.dim_names.index(s) for s in split_dims]
        if type(eval_dims[0]) == str: eval_dims = [self.dim_names.index(e) for e in eval_dims] 
        # If indices not specified, use all.
        if sorted_indices is None: sorted_indices = self.all_sorted_indices
        return np.array(split_dims), np.array(eval_dims), sorted_indices