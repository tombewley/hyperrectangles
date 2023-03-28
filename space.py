from .model import Model
from .tree import Tree
from .node import Node
from .utils import *
import numpy as np
from tqdm import tqdm

class Space:
    """
    Master class for centrally storing data and building models within a continuous space.
    """
    def __init__(self, dim_names, data=None):
        self.dim_names = list(dim_names)
        self.data = data
        # Empty dictionary for storing models.
        self.models = {}; self.fsms = {}

    # Data setter.
    @property
    def data(self): return self._data 
    @data.setter
    def data(self, data):
        if data is None: data = np.empty((0, len(self)))
        assert data.shape[1] == len(self)
        self._data = data
        # Sort data along each dimension up front.
        self.all_sorted_indices = np.argsort(data, axis=0) 
        if data.shape[0]:
            # Scale factors for variance are reciprocals of global variance.
            var = np.var(data, axis=0)
            var[var==0] = 1 # Prevent div/0 error.
            self.global_var_scale = 1 / var
        else: self.global_var_scale = np.zeros(len(self))
            
    # Dunder/magic methods.
    def __repr__(self): return f"{len(self)}D space with {len(self.data)} samples, {len(self.models)} model{'s' if len(self.models) != 1 else ''} and {len(self.fsms)} FSM{'s' if len(self.fsms) != 1 else ''}"
    def __getitem__(self, name): return self.models[name]
    def __len__(self): return len(self.dim_names)

    def subset(self, hr=None, subsample=None):
        """
        Retrieve a subset of the data by per-dimension filtering and/or random subsampling.
        """
        sorted_indices = self.all_sorted_indices
        if hr is not None: sorted_indices = hr_filter_sorted_indices(self, sorted_indices, hr)
        return subsample_sorted_indices(sorted_indices, subsample)

    def tree_depth_first(self, name, split_dims, eval_dims, sorted_indices=None, 
                         max_depth=np.inf, min_samples_leaf=1, corr=False, one_sided=False, pop_power=.5):
        """
        Grow a tree depth-first to max_depth using samples specified by sorted_indices. 
        """
        if corr: assert len(eval_dims) > 1
        split_dims, eval_dims, sorted_indices = self._preflight_check(split_dims, eval_dims, sorted_indices)
        def _recurse(node, depth):
            if node is None: return # This will be the case 50% of the time if doing one-sided.
            if depth < max_depth:
                ok = node._do_greedy_split(split_dims, eval_dims, min_samples_leaf, corr, one_sided, pop_power)
                if ok: _recurse(node.left, depth+1); _recurse(node.right, depth+1)
        root = Node(self, sorted_indices=sorted_indices) 
        _recurse(root, 0)
        self.models[name] = Tree(name, root, split_dims, eval_dims)
        return self.models[name]

    def tree_best_first(self, name, split_dims, eval_dims, sorted_indices=None, 
                        max_num_leaves=np.inf, min_samples_leaf=1, num_from_queue=np.inf, disable_pbar=False): 
        """
        Grow a tree best-first to max_num_leaves using samples specified by sorted_indices. 
        """
        split_dims, eval_dims, sorted_indices = self._preflight_check(split_dims, eval_dims, sorted_indices)
        with tqdm(total=max_num_leaves, disable=disable_pbar) as pbar:
            # Initialise tree with root only.
            root = Node(self, sorted_indices=sorted_indices) 
            self.models[name] = Tree(name, root, split_dims, eval_dims)
            pbar.update(1)
            while len(self.models[name].leaves) < max_num_leaves and len(self.models[name].split_queue) > 0:
                self.models[name].split_next_best(min_samples_leaf, num_from_queue=num_from_queue)
        return self.models[name]

    def model_from_dict(self, name, d):
        """
        Create a flat model from a dictionary object.
        """
        leaves = []
        for node in (d.values() if type(d) == dict else d):
            # Get the maximal (and optionally minimal) hyperrectangle in the correct form.
            hr_max, hr_min = self.listify(node["hr_max"], node["hr_min"] if "hr_min" in node else None,
                                  placeholder=[-np.inf,np.inf], duplicate_singletons=True)  
            # Add a new leaf.
            leaves.append(Node(self, hr_min=hr_min, hr_max=hr_max, meta=node["meta"]))
        self.models[name] = Model(name, leaves)
        return self.models[name]

    def tree_from_dict(self, name, d, redim=None): 
        """
        Create a tree from a dictionary object d. 
        """
        if redim is not None: # "redim" allows renumbering of split dims.
            for v in d.values(): v["split_dim"] = redim[v["split_dim"]]
        def _recurse(node, n): 
            if n in d:
                if not node._do_split(d[n]["split_dim"], split_threshold=d[n]["split_threshold"]):
                    raise ValueError(f"Invalid split threshold for node {n}: \"{d[n]}\".")
                _recurse(node.left, d[n]["left"])
                _recurse(node.right, d[n]["right"])
        root = Node(self, sorted_indices=self.all_sorted_indices)
        _recurse(root, 1) # Root node must have key of 1 in dict.
        split_dims, eval_dims = sorted(set(v["split_dim"] for v in d.values())), [] # NOTE: No eval dims.
        self.models[name] = Tree(name, root, split_dims, eval_dims)
        return self.models[name]

    def tree_from_func(self, name, func):
        """
        Create a tree from a well-formed nested if-then function in Python.
        Tests must use the < or >= operators; split_dims can either be identified with indices, e.g. x[0],
        or with a valid entry in self.dim_names.
        """
        from dill.source import getsource
        lines = [l.strip() for l in getsource(func).split("\n")[:-1]]
        assert lines[0][:3] == "def"
        def _recurse(node, n):
            if lines[n][0] == "#": return _recurse(node, n + 1) 
            elif lines[n][:2] == "if":
                d, o, t = lines[n][3:lines[n].find(":")].split(" ")
                assert o in ("<", ">=")
                try: split_dim = int(d.split("[")[1][:-1]) # If index specified.
                except: split_dim = self.dim_names.index(d) # If dim_name specified.
                split_dims.add(split_dim)
                if not node._do_split(split_dim, split_threshold=float(t)):
                    raise ValueError(f"Invalid split threshold at line {n}: \"{lines[n]}\".")
                n = _recurse(node.left if o == "<" else node.right, n + 1)
                assert lines[n][:lines[n].find(":")] == "else"
                n = _recurse(node.right if o == "<" else node.left, n + 1)
            elif lines[n][:6] == "return":
                # NOTE: Float-convertible return values are stored in node.meta["return"].
                try:    node.meta["return"] = float(lines[n][7:].split(" ")[0])
                except: node.meta["return"] = None
                n += 1
            else: raise ValueError(f"Parse error at line {n}: \"{lines[n]}\".")
            return n
        split_dims, eval_dims = set(), [] # NOTE: No eval dims.
        root = Node(self, sorted_indices=self.all_sorted_indices)
        _recurse(root, 1)
        self.models[name] = Tree(name, root, sorted(split_dims), eval_dims)
        return self.models[name]

    # def fsm_from_model(self, name, model, X, pbar=True):
    #     from .fsm import FSM
    #     self.fsms[name] = FSM(name, model, data=X, pbar=pbar)
    #     return self.fsms[name]

    def idxify(self, *args):
        """
        Dims are convenient to specify as names. 
        This method converts them into numerical indices.
        """
        dims_idx = [] 
        for dims in args:
            if type(dims) in (list, tuple):
                dims = [self.dim_names.index(d) if type(d) != int else d for d in dims]
            elif type(dims) == str: dims = self.dim_names.index(dims) 
            dims_idx.append(dims)
        return dims_idx if len(dims_idx) > 1 else dims_idx[0]

    def listify(self, *args, placeholder=None, duplicate_singletons=False):
        """
        Hyperrectangular sets are convenient to specify as dictionaries.
        This method converts them into lists.
        """
        dim_lists = []
        for x in args:
            if type(x) != dict: dim_list = x # If not a dict, return unchanged.
            else:
                dim_list = [placeholder for _ in range(len(self))]  
                for dim, value in x.items():
                    if duplicate_singletons:
                        try: len(value)
                        except: value = [value, value]
                    dim_list[self.dim_names.index(dim)] = value
            dim_lists.append(dim_list)
        return dim_lists if len(dim_lists) > 1 else dim_lists[0]

    def empty_clone(self):
        return Space(self.dim_names)

    def _preflight_check(self, split_dims, eval_dims, sorted_indices):
        split_dims, eval_dims = self.idxify(split_dims, eval_dims)
        # If indices not specified, use all.
        if sorted_indices is None: sorted_indices = self.all_sorted_indices
        return np.array(split_dims), np.array(eval_dims), sorted_indices
