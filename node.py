from .utils import *
import numpy as np

class Node:
    """
    Class for a node, which is characterised by its members (sorted_indices of data from source), 
    mean, covariance matrix and minimal and maximal bounding boxes. 
    """
    def __init__(self, source, sorted_indices, parent_split_info=None):
        self.source = source # To refer back to the source class.
        self.sorted_indices = sorted_indices
        self.num_samples, num_dims = sorted_indices.shape
        node_data = self.source.data[sorted_indices[:,0]] # Won't actually store this; order doesn't matter.
        self.mean = np.mean(node_data, axis=0)
        if self.num_samples > 1: 
              self.cov = np.cov(node_data, rowvar=False, ddof=0) # ddof=0 overrides bias correction.
        else: self.cov = np.zeros((num_dims, num_dims))
        self.cov_sum = self.cov * self.num_samples
        self.var_sum = np.diag(self.cov_sum)
        # Minimal bounding box.
        self.bb_min = np.array([np.min(node_data, axis=0), np.max(node_data, axis=0)]).T
        # If this node has a parent, use the provided split information to create maximal bounding box.
        if parent_split_info is not None:
            self.bb_max, d, lu, v = parent_split_info # Start by copying parent bb_max.
            self.bb_max[d,lu] = v # Then update one boundary.
        else:
            self.bb_max = np.array([[-np.inf, np.inf] for _ in range(num_dims)]) # If no parent, bb_max is infinite.
        self.split_dim, self.left, self.right, self.gains = None, None, None, {} # To be replaced if and when the node is split.
    
    def _do_greedy_split(self, split_dims, eval_dims, corr=False, one_sided=False, pop_power=.5):
        """
        xxx
        """
        splits, extra = self._find_greedy_splits(split_dims, eval_dims, corr, one_sided, pop_power)
        # Sort by quality and choose the single best.
        split_dim, split_point, qual, index, (left, right) = sorted(splits, key=lambda x: x[2], reverse=True)[0]        
        if qual > 0:
            self.split_dim = split_dim
            # Pick actual numerical value to split at: midpoint of samples either side.
            self.split_value = (self.source.data[left[-1,split_dim],split_dim] + self.source.data[right[0,split_dim],split_dim]) / 2
            if one_sided: # Only create the child for which the split is made.
                self.eval_child_and_dims = index
                do_right = bool(self.eval_child_and_dims[0])
                print(f'Split @ {self.split_dim}={self.split_value} for child {self.eval_child_and_dims[0]} cov({self.source.dim_names[eval_dims[self.eval_child_and_dims[1]]]},{self.source.dim_names[eval_dims[self.eval_child_and_dims[2]]]})')           
            else:
                self.gains['immediate'] = extra
            if (not one_sided) or (not do_right):
                self.left = Node(self.source, left, (self.bb_max.copy(), split_dim, 1, self.split_value))
            if (not one_sided) or do_right:
                self.right = Node(self.source, right, (self.bb_max.copy(), split_dim, 0, self.split_value))
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
            # Evaluate splits along this dim, returning (co)variance sums.
            cov_or_var_sum = self._eval_splits_one_dim(split_dim, eval_dims, cov=corr)
            if corr: 
                # TODO: Fully vectorise this.
                r2 = np.array([np.array([cov_to_r(cov_c_n) # Convert cov to R^2...
                               for cov_c_n in cov_c]) for cov_c in # ...for each child and each num_samples...
                               cov_or_var_sum / n]) # ...where cov is computed by dividing cov_sum by num_samples.          
                # Multiply by population**pop_power to incentivise large populations. 
                # r2_scaled = r2 * n / self.num_samples     
                r2_scaled = r2 * (np.log2(n-1) ** pop_power)
                if one_sided:           
                    # Split quality = maximum value of (R^2 * population**pop_power).
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
                    raise NotImplementedError
                else:
                    # Split quality = sum of reduction in dimensions-scaled variance sums.
                    gain_per_dim = (cov_or_var_sum[1,0] - cov_or_var_sum.sum(axis=0))
                    qual = (gain_per_dim * self.source.scale_factors[eval_dims]).sum(axis=1)
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