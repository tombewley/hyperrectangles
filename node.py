from .utils import *
import numpy as np
import bisect
from sklearn.decomposition import PCA

class Node:
    """
    Class for a node, which is characterised by its samples (sorted_indices of data from space), 
    mean, covariance matrix and minimal and maximal hyperrectangles.
    """
    def __init__(self, space, parent=None, sorted_indices=None, hr_min=None, hr_max=None, meta=None):
        self.space, self.parent = space, parent # "Back-link" to the space and parent node.
        self.hr_max = np.array(hr_max if hr_max is not None else # If a maximal hyperrectangle has been provided, use that.
                      [[-np.inf, np.inf] for _ in self.space.dim_names]) # Otherwise, hr_max is infinite.
        # Split attributes are defined if and when the node is split.
        self.unsplit()
        # This dictionary can be used to store miscellaneous meta information about this node.
        self.meta = {} if meta is None else meta
        # Populate with samples if provided.
        self.populate(sorted_indices, keep_hr_min=False)
        # Overwrite minimal hyperrectangle if provided.
        if hr_min: self.hr_min = np.array(hr_min)
        assert self.hr_min.shape == self.hr_max.shape == (len(self.space), 2)

    # Dunder/magic methods.
    def __repr__(self): return f"Node at {hex(id(self))} with {self.num_samples} samples"
    def __call__(self, *args, **kwargs): return self.membership(*args, **kwargs)
    def __len__(self): return len(self.sorted_indices)
    def __getitem__(self, key): 
        try: return self.__getattribute__(key) # For declared attributes (e.g. self.hr_max).
        except:
            if type(key) == tuple: 
                try: return self.stat(key) # For statistical attributes.
                except: pass
            try: return self.meta[key] # For meta attributes.
            except: return self.data(key) # For data dims.
    def __setitem__(self, key, val): self.meta[key] = val
    def __contains__(self, idx): return idx in self.sorted_indices[:,0] 

    def data(self, *dims): 
        if dims: num_dims = len(dims)
        else: dims = None; num_dims = len(self.space)
        return self.space.data[self.sorted_indices[:,0][:,None], self.space.idxify(dims)].reshape(-1,num_dims)

    def unsplit(self):
        """Remove any split information from this node."""
        self.split_dim, self.split_threshold, self.left, self.right, self.gains = None, None, None, None, {}
    
    def populate(self, sorted_indices, keep_hr_min):
        """
        Populate the node with samples and compute statistics.
        """
        if sorted_indices is None: sorted_indices = np.empty((0, len(self.space)), dtype=int)
        self.sorted_indices = sorted_indices
        self.num_samples, num_dims = sorted_indices.shape
        if self.num_samples > 0: 
            X = self.data() # Won't actually store this; order doesn't matter.
            self.mean = np.mean(X, axis=0)
            # Minimal hyperrectangle is defined by the samples.
            if not keep_hr_min: self.hr_min = np.array([np.min(X, axis=0), np.max(X, axis=0)]).T
            if self.num_samples > 1:
                self.cov = np.cov(X, rowvar=False, ddof=0) # ddof=0 overrides bias correction.                
            else: self.cov = np.zeros((num_dims, num_dims))
        else: 
            self.mean = np.full(num_dims, np.nan)
            self.cov = np.full((num_dims, num_dims), 0.)
            if not keep_hr_min: self.hr_min = np.full((num_dims, 2), np.nan)
        self.cov_sum = self.cov * self.num_samples
        self.var_sum = np.diag(self.cov_sum)      

    def membership(self, x, mode, contain=False):
        """
        Evaluate the membership of x in this node. Each element of x can be None (ignore),
        a scalar (treat as in regular prediction), or a (min, max) interval.
        """
        per_dim = []
        assert len(x) == len(self.space)
        for xd, lims_min, lims_max, mean in zip(x, self.hr_min, self.hr_max, self.mean):
            try:
                # For marginalised (None <=> (-inf, inf) interval).
                if xd is None or np.isnan(xd): 
                    if mode == "fuzzy": per_dim.append(1) 
                # For scalar.
                elif mode == "mean": # Equal to mean.
                    if not xd == mean: return 0 
                elif mode in ("min", "max"): # Inside hyperrectangle.
                    lims = (lims_min if mode == "min" else lims_max)
                    if not(xd >= lims[0] and xd <= lims[1]): return 0
                elif mode == "fuzzy": # Fuzzy membership using both hyperrectangles.
                    to_max_l = xd - lims_max[0]
                    to_max_u = xd - lims_max[1]
                    # Outside hr_max.
                    if not(to_max_l >= 0 and to_max_u <= 0): return 0. 
                    else:
                        to_min_l = xd - lims_min[0]
                        above_min_l = (to_min_l >= 0)
                        to_min_u = xd - lims_min[1]
                        below_min_u = (to_min_u <= 0)
                        # Inside hr_min.
                        if (above_min_l and below_min_u): per_dim.append(1.)
                        # Otherwise (partial membership).
                        else: 
                            # Below lower of hr_min.
                            if not(above_min_l): per_dim.append(to_max_l / (to_max_l - to_min_l))
                            # Above upper of hr_min.
                            else: per_dim.append(to_max_u / (to_max_u - to_min_u))
                else: raise ValueError()
            except:
                # For (min, max) interval.
                if mode == "fuzzy": raise NotImplementedError("Cannot handle intervals in fuzzy mode.")
                elif mode == "mean": # Contains mean.
                    if not (xd[0] <= mean <= xd[1]): return 0 
                elif mode in ("min", "max"): # Intersected/contained by hyperrectangle.
                    lims = (lims_min if mode == "min" else lims_max)
                    compare = [[i >= l for i in xd] for l in lims]
                    if contain:
                        if (not(compare[0][0]) or compare[1][1]): return 0
                    elif (not(compare[0][1]) or compare[1][0]): return 0
        if mode == "fuzzy":
            return abs(min(per_dim)) # Compute total membership using the minimum T-norm.
        return 1

    def stat(self, attr):
        """
        Return a statistical attribute of the data in this node.
        """
        dim = self.space.idxify(attr[1])
        if len(attr) == 3: dim2 = self.space.idxify(attr[2])
        # Mean, standard deviation, or sqrt of covarance (std_c).
        if attr[0] == 'mean': return self.mean[dim]
        if attr[0] == 'var': return self.cov[dim,dim]
        if attr[0] == 'var_sum': return self.var_sum[dim]
        if attr[0] == 'std': return np.sqrt(self.cov[dim,dim])
        if attr[0] == 'std_c': return np.sqrt(self.cov[dim,dim2])
        if attr[0] in ('median','iqr','q1q3'):
            # Median, interquartile range, or lower and upper quartiles.
            q1, q2, q3 = np.quantile(self.space.data[self.sorted_indices[:,dim],dim], (.25,.5,.75))
            if attr[0] == 'median': return q2
            if attr[0] == 'iqr': return q3-q1
            if attr[0] == 'q1q3': return (q1,q3)
        raise ValueError()
    
    def pca(self, dims=None, n_components=None, whiten_by="local"):
        """
        Perform principal component analysis on the data in this node, whitening beforehand
        to ensure that large dimensions do not dominate.
        """
        if dims is None: dims = np.arange(len(self.space))
        else: dims = self.space.idxify(dims)
        X = self.space.data[self.sorted_indices[:,0][:,None],dims]
        if X.shape[0] <= 1: return None, None
        # Whiten data, using either local or global standard deviation.
        mean = X.mean(axis=0)
        std = X.std(axis=0) if whiten_by == 'local' else (1 / (self.space.global_var_scale[dims] ** 0.5))   
        std[std==0] = 1. # Prevent div/0 error.
        X = (X - mean) / std
        # Perform PCA on whitened data.
        pca = PCA(n_components=n_components); pca.fit(X)
        # Return components scaled back by standard deviation, and explained variance ratio.
        # pca.components_ has dimensionality (n_components, len(self.space)), so each component is a row vector.
        return (pca.components_ * std), pca.explained_variance_ratio_

    def feature_importance(self, split_dims, eval_dims):
        """
        Compute local feature importance (var_sum reduction) for all (split_dim, eval_dim) pairs.
        """
        if self.split_dim is None:
            return np.zeros((len(split_dims), len(eval_dims)))
        else:
            fi = self.left.feature_importance(split_dims, eval_dims) + \
                 self.right.feature_importance(split_dims, eval_dims)
            fi[split_dims.index(self.split_dim)] += \
                self.var_sum[eval_dims] - (self.left.var_sum[eval_dims] + self.right.var_sum[eval_dims])
            return fi

    def json(self, *attributes, clip=None): 
        """
        Pack attributes of this node into a dictionary for JSON serialisation.
        """
        assert attributes, "Provide required attributes as arguments"
        d = {}
        for attr in attributes: 
            d[attr] = self[attr] # Make use of __getitem__
            if clip is not None and attr == "hr_max":
                d[attr] = hr_intersect(d[attr], clip) # Clip hr_max to avoid infinite values
            try: d[attr] = d[attr].tolist() # Convert NumPy arrays to lists
            except: pass
        return d

    def _do_split(self, split_dim, split_threshold=None, split_index=None, gains=None):
        """
        Split along split_dim at a specified split_threshold or split_index.
        """
        # Split samples
        if split_threshold is not None: # Threshold -> index
            if not(self.hr_max[split_dim][0] <= split_threshold <= self.hr_max[split_dim][1]): return False
            self.split_threshold = split_threshold
            data = self.space.data[self.sorted_indices[:,split_dim],split_dim]
            split_index = bisect.bisect_left(data, self.split_threshold)
        left, right = split_sorted_indices(self.sorted_indices, split_dim, split_index)
        if split_threshold is None: # Index -> threshold
            self.split_threshold = (self.space.data[left[-1,split_dim],split_dim] + self.space.data[right[0,split_dim],split_dim]) / 2
        self.split_dim = split_dim
        # Split hyperrectangle
        hr_max_left = self.hr_max.copy(); hr_max_left[self.split_dim,1] = self.split_threshold
        hr_max_right = self.hr_max.copy(); hr_max_right[self.split_dim,0] = self.split_threshold
        # Make children
        self.left = Node(self.space, parent=self, sorted_indices=left, hr_max=hr_max_left)
        self.right = Node(self.space, parent=self, sorted_indices=right, hr_max=hr_max_right)
        # Store gains
        if gains is not None: self.gains["immediate"] = gains
        return True

    def _find_greedy_split(self, split_finder, split_dims, eval_dims, min_samples_leaf, entropy, store_all_qual):
        """
        Find all greedy splits for split_dims, using split_finder and eval_dims,
        then choose a single split_dim using a softmax or hard argmax decision.
        """
        # Only attempt to split if there are enough samples.
        if len(self) >= 2*min_samples_leaf:
            splits, gains = split_finder(self, split_dims, eval_dims, min_samples_leaf, store_all_qual)
            if splits:
                if entropy > 0.:
                    # Choose split dimension using softmax with entropy coefficient.
                    q = np.array([x[2] for x in splits])
                    q_max = q.max()
                    q_norm = (q / (q_max if q_max > 0. else 1.)) - 1. # Prevents overflow
                    exp_q = np.exp(q_norm / entropy)
                    p = exp_q / exp_q.sum()
                    split_dim, split_index, qual = splits[np.random.choice(range(len(splits)), p=p)]
                else:
                    # Sort splits by quality and choose the single best.
                    split_dim, split_index, qual = sorted(splits, key=lambda x: x[2], reverse=True)[0]
                return split_dim, split_index, qual, gains
        return None, None, -np.inf, None 
