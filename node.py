from .utils import *
import numpy as np
import bisect
from sklearn.decomposition import PCA

class Node:
    """
    Class for a node, which is characterised by its members (sorted_indices of data from source), 
    mean, covariance matrix and minimal and maximal bounding boxes. 
    """
    def __init__(self, source, sorted_indices, bb_max=None, parent_split_info=None):
        self.source = source # To refer back to the source class.
        self.sorted_indices = sorted_indices
        self.num_samples, num_dims = sorted_indices.shape
        X = self.source.data[sorted_indices[:,0]] # Won't actually store this; order doesn't matter.
        if self.num_samples > 0: 
            self.mean = np.mean(X, axis=0)
            # Minimal bounding box is defined by the samples.
            self.bb_min = np.array([np.nanmin(X, axis=0), np.nanmax(X, axis=0)]).T
            if self.num_samples > 1:
                self.cov = np.cov(X, rowvar=False, ddof=0) # ddof=0 overrides bias correction.                
        else: 
            self.mean = np.full(num_dims, np.nan)
            self.bb_min = np.full((num_dims, 2), np.nan)
        try: self.cov
        except: self.cov = np.zeros((num_dims, num_dims))
        self.cov_sum = self.cov * self.num_samples
        self.var_sum = np.diag(self.cov_sum)
        if bb_max:
            # If a maximal bounding box has been provided, just use that.
            self.bb_max = bb_max
        elif parent_split_info:
            # If this node has a parent, use the provided split information to create bb_max.
            self.bb_max, d, lu, v = parent_split_info; self.bb_max[d,lu] = v 
        else:
            # Otherwise, bb_max is infinite.
            self.bb_max = np.array([[-np.inf, np.inf] for _ in range(num_dims)]) 
        # These attributes are defined if and when the node is split.
        self.split_dim, self.split_threshold, self.left, self.right, self.gains = None, None, None, None, {} 
        # This dictionary can be used to store miscellaneous meta information about this node.
        self.meta = {}

    def attr(self, attr):
        """
        Compute a statistical attribute for this node.
        """
        # Allow dim_name to be specified instead of number.
        if type(attr[1]) == str: dim = self.source.dim_names.index(attr[1])
        if len(attr) == 3 and type(attr[2]) == str: dim2 = self.source.dim_names.index(attr[2])
        # Mean, standard deviation, or sqrt of covarance (std_c).
        if attr[0] == 'mean': return self.mean[dim]
        elif attr[0] == 'std': return np.sqrt(self.cov[dim,dim])
        elif attr[0] == 'std_c': return np.sqrt(self.cov[dim,dim2])
        elif attr[0] in ('median','iqr','q1q3'):
            # Median, interquartile range, or lower and upper quartiles.
            q1, q2, q3 = np.quantile(self.source.data[self.sorted_indices[:,dim],dim], (.25,.5,.75))
            if attr[0] == 'median': return q2
            elif attr[0] == 'iqr': return q3-q1
            elif attr[0] == 'q1q3': return (q1,q3)
    
    def pca(self, dims=None, n_components=None, whiten_by='local'):
        """
        Perform principal component analysis on the data at this node, whitening beforehand
        to ensure that large dimensions do not dominate.
        """
        X = self.source.data[self.sorted_indices[:,0][:,None],dims]
        if X.shape[0] == 1: return None, None
        if dims is None: dims = np.arange(len(self.source.dim_names))
        # Allow dim_names to be specified instead of numbers.
        if type(dims[0]) == str: dims = [self.source.dim_names.index(d) for d in dims]
        # Whiten data, using either local or global standard deviation.
        mean = X.mean(axis=0)
        std = X.std(axis=0) if whiten_by == 'local' else (1 / (self.source.global_var_scale[dims] ** 0.5))   
        X = (X - mean) / std
        # Perform PCA on whitened data.
        pca = PCA(n_components=n_components); pca.fit(X)
        # Return components scaled back by standard deviation, and explained variance ratio.
        return (pca.components_ * std), pca.explained_variance_ratio_
    
    def _do_manual_split(self, split_dim, split_threshold):
        """
        Split using a manually-defined split_dim and split_threshold.
        """
        if not(self.bb_max[split_dim][0] <= split_threshold <= self.bb_max[split_dim][1]): return False
        self.split_dim, self.split_threshold = split_dim, split_threshold
        data = self.source.data[self.sorted_indices[:,self.split_dim], self.split_dim]
        split_index = bisect.bisect(data, self.split_threshold)
        left, right = split_sorted_indices(self.sorted_indices, self.split_dim, split_index)
        self.left = Node(self.source, left, parent_split_info=(self.bb_max.copy(), self.split_dim, 1, self.split_threshold))
        self.right = Node(self.source, right, parent_split_info=(self.bb_max.copy(), self.split_dim, 0, self.split_threshold))
        return True

    def _do_greedy_split(self, split_dims, eval_dims, corr=False, one_sided=False, pop_power=.5):
        """
        Find and implement the greediest split given split_dims and eval_dims.
        """
        splits, extra = self._find_greedy_splits(split_dims, eval_dims, corr, one_sided, pop_power)
        if splits:
            # Sort by quality and choose the single best.
            split_dim, split_point, qual, index, (left, right) = sorted(splits, key=lambda x: x[2], reverse=True)[0]        
            if qual > 0:
                self.split_dim = split_dim
                # Pick actual numerical threshold to split at: midpoint of samples either side.
                self.split_threshold = (self.source.data[left[-1,split_dim],split_dim] + self.source.data[right[0,split_dim],split_dim]) / 2
                if one_sided: # Only create the child for which the split is made.
                    self.eval_child_and_dims = index
                    do_right = bool(self.eval_child_and_dims[0])
                    print(f'Split @ {self.split_dim}={self.split_threshold} for child {self.eval_child_and_dims[0]} cov({self.source.dim_names[eval_dims[self.eval_child_and_dims[1]]]},{self.source.dim_names[eval_dims[self.eval_child_and_dims[2]]]})')           
                else:
                    self.gains['immediate'] = extra
                if (not one_sided) or (not do_right):
                    self.left = Node(self.source, left, parent_split_info=(self.bb_max.copy(), split_dim, 1, self.split_threshold))
                if (not one_sided) or do_right:
                    self.right = Node(self.source, right, parent_split_info=(self.bb_max.copy(), split_dim, 0, self.split_threshold))
                return True, extra
        return False, extra

    def _find_greedy_splits(self, split_dims, eval_dims, corr=False, one_sided=False, pop_power=.5):
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
            # Cannot split on a dim if there is no variance, so skip.
            if self.var_sum[split_dim] == 0: continue
            # Evaluate splits along this dim, returning (co)variance sums.
            cov_or_var_sum = self._eval_splits_one_dim(split_dim, eval_dims, cov=corr)
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
                    right, split_point, d1, d2 = np.unravel_index(np.nanargmax(r2_scaled), r2_scaled.shape)
                    qual_max = r2_scaled[(right, split_point, d1, d2)] - r2_scaled[(1, 0, d1, d2)]
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
                    gain_per_dim = (cov_or_var_sum[1,0] - cov_or_var_sum.sum(axis=0))
                    qual = (gain_per_dim * self.source.global_var_scale[eval_dims]).sum(axis=1)
                    split_point = np.argmax(qual) # Greedy split is the one with the highest quality.                    
                    qual_max = qual[split_point]
                    extra.append(gain_per_dim[split_point]) # Extra = gain_per_dim at split point.
            # Store split info.
            splits.append((split_dim, split_point, qual_max, (right, d1, d2) if corr else None,
                           split_sorted_indices(self.sorted_indices, split_dim, split_point)))
        return splits, np.array(extra)

    def _eval_splits_one_dim(self, split_dim, eval_dims, cov=False):
        """
        Try splitting the node along one split_dim, calculating (co)variance sums along eval_dims.  
        """
        eval_data = self.source.data[self.sorted_indices[:,split_dim][:,None],eval_dims] # Gather data from source class.
        d = len(eval_dims)
        mean = np.zeros((2,self.num_samples,d))
        if cov: # For full covariance matrix.
            cov_sum = np.zeros((2,self.num_samples,d,d))
            cov_sum[1,0] = self.cov_sum[eval_dims[:,None],eval_dims]
        else: # Just variances (diagonal of cov).
            var_sum = mean.copy()
            var_sum[1,0] = self.var_sum[eval_dims]
        mean[1,0] = self.mean[eval_dims] 
        for num_left in range(1,self.num_samples): 
            num_right = self.num_samples - num_left
            x = eval_data[num_left-1]
            if cov:
                mean[0,num_left], cov_sum[0,num_left] = increment_mean_and_cov_sum(num_left,  mean[0,num_left-1], cov_sum[0,num_left-1], x, 1)
                mean[1,num_left], cov_sum[1,num_left] = increment_mean_and_cov_sum(num_right, mean[1,num_left-1], cov_sum[1,num_left-1], x, -1)
            else:
                mean[0,num_left], var_sum[0,num_left] = increment_mean_and_var_sum(num_left,  mean[0,num_left-1], var_sum[0,num_left-1], x, 1)
                mean[1,num_left], var_sum[1,num_left] = increment_mean_and_var_sum(num_right, mean[1,num_left-1], var_sum[1,num_left-1], x, -1)            
        return cov_sum if cov else var_sum