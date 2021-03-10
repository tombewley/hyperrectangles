from .utils import *
import numpy as np
import bisect
from sklearn.decomposition import PCA

class Node:
    """
    Class for a node, which is characterised by its samples (sorted_indices of data from space), 
    mean, covariance matrix and minimal and maximal bounding boxes. 
    """
    def __init__(self, space, sorted_indices=None, bb_min=None, bb_max=None, meta={}):
        self.space = space # To refer back to the space class.     
        self.bb_max = np.array(bb_max if bb_max is not None else # If a maximal bounding box has been provided, use that.
                      [[-np.inf, np.inf] for _ in self.space.dim_names]) # Otherwise, bb_max is infinite.
        # These attributes are defined if and when the node is split.
        self.split_dim, self.split_threshold, self.left, self.right, self.gains = None, None, None, None, {} 
        # This dictionary can be used to store miscellaneous meta information about this node.
        self.meta = meta
        # Populate with samples if provided.
        self.populate(sorted_indices, keep_bb_min=False)
        # Overwrite minimal bounding box if provided.
        if bb_min: self.bb_min = np.array(bb_min)
        assert self.bb_min.shape == self.bb_max.shape == (len(self.space), 2)

    # Dunder/magic methods.
    def __repr__(self): return f"Node with {self.num_samples} samples"
    def __call__(self, *args, **kwargs): return self.membership(*args, **kwargs)
    def __len__(self): return len(self.sorted_indices)
    # def __getattr__(self, key): return self.__getitem__(key)
    def __getitem__(self, key): 
        try: return self.__getattribute__(key) # For declared attributes (e.g. self.bb_max).
        except:
            if type(key) == tuple: 
                try: return self.stat(key) # For statistical attributes.
                except: pass
            return self.meta[key] # For meta attributes.
    def __setitem__(self, key, val): self.meta[key] = val
    def __contains__(self, idx): return idx in self.sorted_indices[:,0] 

    def data(self, *dims): 
        if dims: num_dims = len(dims)
        else: dims = None; num_dims = len(self.space)
        return self.space.data[self.sorted_indices[:,0][:,None], self.space.idxify(dims)].reshape(-1,num_dims)
    
    def populate(self, sorted_indices, keep_bb_min):
        """
        Populate the node with samples and compute statistics.
        """
        if sorted_indices is None: sorted_indices = np.empty((0, len(self.space)))
        self.sorted_indices = sorted_indices
        self.num_samples, num_dims = sorted_indices.shape
        if self.num_samples > 0: 
            X = self.data() # Won't actually store this; order doesn't matter.
            self.mean = np.mean(X, axis=0)
            # Minimal bounding box is defined by the samples.
            if not keep_bb_min: self.bb_min = np.array([np.min(X, axis=0), np.max(X, axis=0)]).T
            if self.num_samples > 1:
                self.cov = np.cov(X, rowvar=False, ddof=0) # ddof=0 overrides bias correction.                
        else: 
            self.mean = np.full(num_dims, np.nan)
            if not keep_bb_min: self.bb_min = np.full((num_dims, 2), np.nan)
        try: self.cov
        except: self.cov = np.zeros((num_dims, num_dims))
        self.cov_sum = self.cov * self.num_samples
        self.var_sum = np.diag(self.cov_sum)      

    def membership(self, x, mode, contain=False):
        """
        Evaluate the membership of x in this node. Each element of x can be None (ignore),
        a scalar (treat as in regular prediction), or a (min, max) interval.
        """
        per_dim = []
        assert len(x) == len(self.space)
        for xd, lims_min, lims_max, mean in zip(x, self.bb_min, self.bb_max, self.mean):
            try:
                # For marginalised (None <=> (-inf, inf) interval).
                if xd is None or np.isnan(xd): 
                    if mode == "fuzzy": per_dim.append(1) 
                # For scalar.
                elif mode == "mean": # Equal to mean.
                    if not xd == mean: return 0 
                elif mode in ("min", "max"): # Inside bounding box.
                    lims = (lims_min if mode == "min" else lims_max)
                    if not(xd >= lims[0] and xd <= lims[1]): return 0
                elif mode == "fuzzy": # Fuzzy membership using both bounding boxes.
                    to_max_l = xd - lims_max[0]
                    to_max_u = xd - lims_max[1]
                    # Outside bb_max.
                    if not(to_max_l >= 0 and to_max_u <= 0): return 0. 
                    else:
                        to_min_l = xd - lims_min[0]
                        above_min_l = (to_min_l >= 0)
                        to_min_u = xd - lims_min[1]
                        below_min_u = (to_min_u <= 0)
                        # Inside bb_min.
                        if (above_min_l and below_min_u): per_dim.append(1.)
                        # Otherwise (partial membership).
                        else: 
                            # Below lower of bb_min.
                            if not(above_min_l): per_dim.append(to_max_l / (to_max_l - to_min_l))
                            # Above upper of bb_min.
                            else: per_dim.append(to_max_u / (to_max_u - to_min_u))
                else: raise ValueError()
            except:
                # For (min, max) interval.
                if mode == "fuzzy": raise NotImplementedError("Cannot handle intervals in fuzzy mode.")
                elif mode == "mean": # Contains mean.
                    if not (xd[0] <= mean <= xd[1]): return 0 
                elif mode in ("min", "max"): # Intersected/contained by bounding box.
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

    def json(self, *attributes, clip=None): 
        """
        Pack attributes of this node into a dictionary for JSON serialisation.
        """
        assert attributes, "Provide required attributes as arguments"
        d = {}
        for attr in attributes: 
            d[attr] = self[attr] # Make use of __getitem__
            if clip is not None and attr == "bb_max": 
                d[attr] = bb_clip(d[attr], clip) # Clip bb_max to avoid infinite values
            try: d[attr] = d[attr].tolist() # Convert NumPy arrays to lists
            except: pass
        return d

    def _do_manual_split(self, split_dim, split_threshold=None, split_index=None):
        """
        Split using a manually-defined split_dim and split_threshold or split_index.
        """
        if split_threshold is not None:
            if not(self.bb_max[split_dim][0] <= split_threshold <= self.bb_max[split_dim][1]): return False
            self.split_dim, self.split_threshold = split_dim, split_threshold
            # Split samples.
            data = self.space.data[self.sorted_indices[:,self.split_dim], self.split_dim]
            split_index = bisect.bisect(data, self.split_threshold)
            left, right = split_sorted_indices(self.sorted_indices, self.split_dim, split_index)
        else:
            self.split_dim = split_dim
            left, right = split_sorted_indices(self.sorted_indices, self.split_dim, split_index)
            self.split_threshold = (self.space.data[left[-1,split_dim],split_dim] + self.space.data[right[0,split_dim],split_dim]) / 2
        # Split bounding box.
        bb_max_left = self.bb_max.copy(); bb_max_left[self.split_dim,1] = self.split_threshold
        bb_max_right = self.bb_max.copy(); bb_max_right[self.split_dim,0] = self.split_threshold
        # Make children.
        self.left = Node(self.space, sorted_indices=left, bb_max=bb_max_left)
        self.right = Node(self.space, sorted_indices=right, bb_max=bb_max_right)
        return True

    def _do_greedy_split(self, split_dims, eval_dims, min_samples_leaf, corr=False, one_sided=False, pop_power=.5):
        """
        Find and implement the greediest split given split_dims and eval_dims.
        """
        # Only attempt to split if there are enough samples.
        if len(self) >= 2*min_samples_leaf:
            splits, extra = self._find_greedy_splits(split_dims, eval_dims, min_samples_leaf, corr, one_sided, pop_power)
            if splits:
                # Sort splits by quality and choose the single best.
                split_dim, split_index, qual, index = sorted(splits, key=lambda x: x[2], reverse=True)[0]        
                if qual > 0:
                    self.split_dim = split_dim
                    # Split sorted indices at index.
                    left, right = split_sorted_indices(self.sorted_indices, self.split_dim, split_index)
                    # Compute numerical threshold to split at: midpoint of samples either side.
                    self.split_threshold = (self.space.data[left[-1,split_dim],split_dim] + self.space.data[right[0,split_dim],split_dim]) / 2
                    if one_sided: # Only create the child for which the split is made.
                        self.eval_child_and_dims = index
                        do_right = bool(self.eval_child_and_dims[0])
                        print(f'Split @ {self.split_dim}={self.split_threshold} for child {self.eval_child_and_dims[0]} cov({self.space.dim_names[eval_dims[self.eval_child_and_dims[1]]]},{self.space.dim_names[eval_dims[self.eval_child_and_dims[2]]]})')
                    else: self.gains["immediate"] = extra
                    # Split bounding box and make children.
                    if (not one_sided) or (not do_right):
                        bb_max_left = self.bb_max.copy(); bb_max_left[self.split_dim,1] = self.split_threshold
                        self.left = Node(self.space, sorted_indices=left, bb_max=bb_max_left)
                    if (not one_sided) or do_right:
                        bb_max_right = self.bb_max.copy(); bb_max_right[self.split_dim,0] = self.split_threshold
                        self.right = Node(self.space, sorted_indices=right, bb_max=bb_max_right)
                    return True
        return False

    def _find_greedy_splits(self, split_dims, eval_dims, min_samples_leaf, corr, one_sided, pop_power):
        """
        Try splitting the node along several split_dims, measuring quality using eval_dims.  
        Return the best split from each dim.
        """
        if corr:
            # Sequences of num_samples for left and right children.
            n = np.arange(self.num_samples)
            n = np.vstack((n, np.flip(n+1)))[:,:,None,None]
        splits, extra = [], []
        for split_dim in split_dims:
            # Apply two kinds of constraint to the split point:
            #   (1) Must be a "threshold" point where the samples either side do not have equal values.
            valid_split_indices = np.unique(self.space.data[self.sorted_indices[:,split_dim][:,None],split_dim], return_index=True)[1]
            #   (2) Must obey min_samples_leaf.
            valid_split_indices = [s for s in valid_split_indices if s >= min_samples_leaf and s <= self.num_samples-min_samples_leaf]
            # Cannot split on a dim if there are no valid split points, so skip.
            if valid_split_indices == []: extra.append(np.nan); continue
            # Evaluate splits along this dim, returning (co)variance sums.
            cov_or_var_sum = self._eval_splits_one_dim(split_dim, eval_dims, min_samples_leaf, cov=corr)
            if corr: 
                # TODO: Fully vectorise this.
                r2 = np.array([np.array([cov_to_r2(cov_c_n) # Convert cov to R^2...
                               for cov_c_n in cov_c]) for cov_c in # ...for each child and each num_samples...
                               cov_or_var_sum / n]) # ...where cov is computed by dividing cov_sum by num_samples.          
                # Scaling incentivises large populations. 
                r2_scaled = r2 * (np.log2(n-1) ** pop_power)
                # r2_scaled = r2 * n / self.num_samples     
                if one_sided:           
                    # Split quality = maximum value of (R^2 * log2(population-1)**pop_power).
                    right, split_index, d1, d2 = np.unravel_index(np.nanargmax(r2_scaled), r2_scaled.shape)
                    qual_max = r2_scaled[(right, split_index, d1, d2)] - r2_scaled[(1, 0, d1, d2)]
                    extra.append(r2_scaled) # Extra = r2_scaled at all points.
                else:
                    pca = [[np.linalg.eig(cov_c_n) if not np.isnan(cov_c_n).any() else None
                           for cov_c_n in cov_c] for cov_c in 
                           cov_or_var_sum / n]
                    return pca
            else:    
                if one_sided: 
                    raise NotImplementedError()
                else:
                    # Split quality = sum of reduction in dimensions-scaled variance sums.
                    gain_per_dim = (cov_or_var_sum[1,0] - 
                                    cov_or_var_sum[:,valid_split_indices,:].sum(axis=0))
                    qual = (gain_per_dim * self.space.global_var_scale[eval_dims]).sum(axis=1)
                    # Greedy split is the one with the highest quality.
                    greedy = np.argmax(qual)      
                    split_index = valid_split_indices[greedy]   
                    qual_max = qual[greedy]
                    extra.append(gain_per_dim[greedy]) # Extra = gain_per_dim at split point.
            # Store split info.
            splits.append((split_dim, split_index, qual_max, (right, d1, d2) if corr else None))
        return splits, np.array(extra)

    def _eval_splits_one_dim(self, split_dim, eval_dims, min_samples_leaf, cov=False):
        """
        Try splitting the node along one split_dim, calculating (co)variance sums along eval_dims.  
        """
        eval_data = self.space.data[self.sorted_indices[:,split_dim][:,None],eval_dims] 
        d = len(eval_dims)
        mean = np.zeros((2,self.num_samples+1,d))
        if cov: # For full covariance matrix.
            cov_sum = np.zeros((2,self.num_samples+1,d,d))
            cov_sum[1,0] = self.cov_sum[eval_dims[:,None],eval_dims]
        else: # Just variances (diagonal of cov).
            var_sum = mean.copy()
            var_sum[1,0] = self.var_sum[eval_dims]
        mean[1,0] = self.mean[eval_dims] 
        for num_left in range(1,(self.num_samples+1)-min_samples_leaf): 
            num_right = self.num_samples - num_left
            x = eval_data[num_left-1]
            if cov:
                mean[0,num_left], cov_sum[0,num_left] = increment_mean_and_cov_sum(num_left,  mean[0,num_left-1], cov_sum[0,num_left-1], x, 1)
                mean[1,num_left], cov_sum[1,num_left] = increment_mean_and_cov_sum(num_right, mean[1,num_left-1], cov_sum[1,num_left-1], x, -1)
            else:
                mean[0,num_left], var_sum[0,num_left] = increment_mean_and_var_sum(num_left,  mean[0,num_left-1], var_sum[0,num_left-1], x, 1)
                mean[1,num_left], var_sum[1,num_left] = increment_mean_and_var_sum(num_right, mean[1,num_left-1], var_sum[1,num_left-1], x, -1)            
        return cov_sum if cov else var_sum

# ================================================================

    def _eval_splits_one_dim_transition(self, split_dim, sim_dim, succ_leaf_all, sim_params):
        """
        Try splitting the node along one dim using the transition impurity method.
        """
        indices = self.sorted_indices[:,split_dim]
        sim_data = [None for _ in indices] if sim_dim is None else self.space.data[indices,sim_dim]
        succ_leaf = [succ_leaf_all[idx] if succ_leaf_all[idx] != self else 1 for idx in indices] # 1 is placeholder for right, 0 is placeholder for left.        
        indices_split = [set(), set(indices)]
        imp_sum = np.zeros((2,self.num_samples+1))
        imp_sum[1,0] = self.t_imp_sum        
        for num_left in range(1,self.num_samples+1):
            print(num_left)
            idx, x, s = indices[num_left-1], sim_data[num_left-1], succ_leaf[num_left-1]
            indices_split[0].add(idx); indices_split[1].remove(idx) # Transfer index from left to right.
            imp_sum[:,num_left] = imp_sum[:,num_left-1] # Copy over previous impurities for incremental.
            for l_or_r in (0,1): # Left or right.
                x_l_or_r = sim_data[:num_left] if l_or_r == 0 else sim_data[num_left:]
                succ_leaf_l_or_r = succ_leaf[:num_left] if l_or_r == 0 else succ_leaf[num_left:]
                # Compute contributition to impurity_sum.
                contrib = 2*transition_imp_contrib(x, s, x_l_or_r, succ_leaf_l_or_r, sim_params) # Multiply x2 due to symmetry.         
                imp_sum[l_or_r,num_left] += contrib if l_or_r == 0 else -contrib # Add for left, subtract for right.
                if idx-1 in indices_split[l_or_r]: # May need to correct impurity contribution of predecessor. 
                    loc_p = np.where(indices == idx-1)[0][0]
                    if succ_leaf[loc_p] is not None:
                        # Correction is a three-step process:
                        imp_sum[l_or_r,num_left] -= 2*transition_imp_contrib(sim_data[loc_p], succ_leaf[loc_p], x_l_or_r, succ_leaf_l_or_r, sim_params) # Remove old...
                        succ_leaf[loc_p] = 0 # ...update...
                        succ_leaf_l_or_r = succ_leaf[:num_left] if l_or_r == 0 else succ_leaf[num_left:]
                        imp_sum[l_or_r,num_left] += 2*transition_imp_contrib(sim_data[loc_p], succ_leaf[loc_p], x_l_or_r, succ_leaf_l_or_r, sim_params) # ...add new.
        return imp_sum