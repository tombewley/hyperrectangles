import numpy as np
import bisect
from itertools import product
import numba

# ===============================
# OPERATIONS USED FOR VARIANCE-BASED SPLITTING

def variance_based_split_finder(node, split_dims, eval_dims, min_samples_leaf, store_all_qual=False):
    """
    Identify and evaluate all valid splits of node along the dimensions of split_data, incrementally calculating variance sums within eval_data.
    Calculate split quality = sum of reduction in dimension-scaled variance sums and find the greedy split along each dim.
    """
    # Gather attributes from the node
    parent_mean = node.mean[eval_dims]
    parent_var_sum = node.var_sum[eval_dims]
    var_scale = node.space.global_var_scale[eval_dims]
    split_data = node.space.data[node.sorted_indices[:,split_dims],split_dims]
    eval_data = node.space.data[node.sorted_indices[:,split_dims][:,:,None],eval_dims]
    # Call jitted inner function
    all_qual, split_indices = _vbsf_inner(split_data, eval_data, min_samples_leaf, parent_mean, parent_var_sum, var_scale)
    splits = []
    for split_dim, split_index in zip(split_dims, split_indices):
        if split_index >= 0: # NOTE: Default is -1 if no valid_split_indices
            splits.append((split_dim, split_index, all_qual[split_index,split_dim]))
    # If applicable, store all split thresholds and quality values
    if store_all_qual:
        node.all_split_thresholds, node.all_qual = {}, {}
        for d in range(len(split_dims)):
            node.all_split_thresholds[split_dims[d]] = (split_data[:-1,d] + split_data[1:,d]) / 2
            node.all_qual[split_dims[d]] = all_qual[1:,d]
    return splits, np.array([]) # NOTE: Greedy gains not implemented

@numba.jit(nopython=True, cache=True, parallel=True)
def _vbsf_inner(split_data, eval_data, min_samples_leaf, parent_mean, parent_var_sum, var_scale):
    """
    Jitted inner function for variance_based_split_finder.
    """
    def increment_mean_and_var_sum(n, mean, var_sum, x, sign):
        """
        Welford's online algorithm for incremental sum-of-variance computation,
        adapted from https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
        """
        d_last = x - mean
        mean = mean + (sign * (d_last / n)) # Can't do += because this modifies the NumPy array in place!
        d = x - mean
        var_sum = var_sum + (sign * (d_last * d))
        return mean, np.maximum(var_sum, 0) # Clip at zero

    num_samples, num_split_dims = split_data.shape
    all_qual = np.full_like(split_data, np.nan)
    greedy_split_indices = np.full(num_split_dims, -1, dtype=np.int32)
    # greedy_gains = [] # NOTE: Not implemented
    for d in numba.prange(num_split_dims): # TODO: Faster to vectorise the entire process along d?
        # Apply two kinds of constraint to the split points:
        #   (1) Must be a "threshold" point where the samples either side do not have equal values
        valid_split_indices = np.where(split_data[1:,d] - split_data[:-1,d])[0] + 1 # NOTE: 0 will not be included
        #   (2) Must obey min_samples_leaf
        mask = np.logical_and(valid_split_indices >= min_samples_leaf, valid_split_indices <= num_samples-min_samples_leaf)
        valid_split_indices = valid_split_indices[mask]
        # Cannot split on a dim if there are no valid split points
        if len(valid_split_indices) == 0: continue
        # greedy_gains.append(np.full(len(eval_dims), np.nan)) # NOTE: Not implemented
        max_num_left = valid_split_indices[-1] + 1 # +1 needed
        mean = np.zeros((2, max_num_left, eval_data.shape[2]))
        var_sum = mean.copy()
        mean[1,0] = parent_mean
        var_sum[1,0] = parent_var_sum
        for num_left in range(1, max_num_left): # Need to start at 1 for incremental calculation to work
            num_right = num_samples - num_left
            x = eval_data[num_left-1,d]
            mean[0,num_left], var_sum[0,num_left] = increment_mean_and_var_sum(num_left,  mean[0,num_left-1], var_sum[0,num_left-1], x, 1)
            mean[1,num_left], var_sum[1,num_left] = increment_mean_and_var_sum(num_right, mean[1,num_left-1], var_sum[1,num_left-1], x, -1)
        gains = var_sum[1,0] - var_sum[:,valid_split_indices,:].sum(axis=0)
        all_qual[valid_split_indices,d] = (gains * var_scale).sum(axis=1)
        # Greedy split is the one with the highest quality
        greedy = np.argmax(all_qual[valid_split_indices,d])
        greedy_split_indices[d] = valid_split_indices[greedy]
        # greedy_gains.append(gains_this_dim[greedy]) # NOTE: Not implemented
    return all_qual, greedy_split_indices

# ===============================
# OPERATIONS ON SORTED INDICES

def split_sorted_indices(sorted_indices, split_dim, split_index):
    """
    Split a sorted_indices array at a point along one of the dimensions,
    preserving the order along all others.
    """
    dims = range(sorted_indices.shape[1])
    left, right = [[] for _ in dims], [[] for _ in dims]
    left[split_dim], right[split_dim] = np.split(sorted_indices[:,split_dim], [split_index])
    for other_dim in dims:
        put_left = np.in1d(sorted_indices[:,other_dim], left[split_dim])
        left[other_dim] = sorted_indices[put_left, other_dim]
        right[other_dim] = sorted_indices[np.logical_not(put_left), other_dim]
    return np.array(left).T, np.array(right).T

# NOTE: Not working; getting an int32/int64 error
@numba.jit(nopython=True, cache=True, parallel=True) # Python needed for set
# def split_sorted_indices(sorted_indices, split_dim, split_index):
#     """
#     Split a sorted_indices array at a point along one dimension, preserving the order along all.
#     """
#     num_samples = sorted_indices.shape[0]
#     num_dims = sorted_indices.shape[1]
#     left = np.empty((split_index, num_dims), dtype=sorted_indices.dtype)
#     right = np.empty((num_samples-split_index, num_dims), dtype=sorted_indices.dtype)
#     left[:,split_dim], right[:,split_dim] = np.split(sorted_indices[:,split_dim], [split_index])
#     left_indices = set(left[:,split_dim])
#     for d in numba.prange(num_dims):
#         if d == split_dim: continue
#         left_mask = np.full(num_samples, False)
#         for i in numba.prange(num_samples):
#             if sorted_indices[i,d] in left_indices: left_mask[i] = True
#         left[:,d], right[:,d] = sorted_indices[left_mask,d], sorted_indices[~left_mask,d]
#     return left, right

def hr_filter_sorted_indices(space, sorted_indices, hr):
    """
    Making use of the split_sorted_indices function, filter sorted_indices using a hr.
    Allow hr to be specified as a dict.
    """
    for split_dim, lims in enumerate(space.listify(hr)):
        if lims is None: continue # If nothing specified for this lim.
        for lu, lim in enumerate(lims):
            if np.isfinite(lim):
                data = space.data[sorted_indices[:,split_dim], split_dim] # Must reselect each time.
                if lu == 0:
                    # For lower limit, bisect to the right.
                    split_index = bisect.bisect_right(data, lim)
                    _, sorted_indices = split_sorted_indices(sorted_indices, split_dim, split_index)
                else:
                    # For upper limit, bisect to the left.
                    split_index = bisect.bisect_left(data, lim)
                    sorted_indices, _ = split_sorted_indices(sorted_indices, split_dim, split_index)    
    return sorted_indices

def subsample_sorted_indices(sorted_indices, size):
    """
    Subsample a sorted_indices array, preserving order.
    """
    if size is None or size >= len(sorted_indices): return sorted_indices
    dims = range(sorted_indices.shape[1])
    subset_indices = np.random.choice(sorted_indices[:,0], replace=False, size=size)
    subset = [[] for _ in dims]
    for dim in dims:
        keep = np.in1d(sorted_indices[:,dim], subset_indices)
        subset[dim] = sorted_indices[keep, dim]
    return np.array(subset).T

def dataframe(space, sorted_indices, index_col):
    """
    Convert a set of sorted indices into a Pandas dataframe.
    """
    import pandas as pd
    return pd.DataFrame(space.data[sorted_indices[:,0]], columns=space.dim_names).set_index(index_col)

def group_along_dim(space, dim):
    """
    Create a list by grouping the data along dim, then sub-sorting the indices.
    """
    dim = space.idxify(dim)
    sorted_indices = space.all_sorted_indices[:,dim]
    data_grouped, pos_last = [], 0
    for pos, idx in enumerate(sorted_indices):
        dim_value = space.data[idx,dim]
        if pos == 0: dim_value_last = dim_value
        elif dim_value != dim_value_last:
            data_grouped.append(space.data[np.sort(sorted_indices[pos_last:pos])]) 
            pos_last, dim_value_last = pos, dim_value
    data_grouped.append(space.data[np.sort(sorted_indices[pos_last:])]) 
    return data_grouped    

# ===============================
# OPERATIONS ON BOUNDING BOXES

def hr_intersect(hr_a, hr_b):
    """
    Find intersection between two hyperrectangles.
    """
    l = np.maximum(hr_a[:,0], hr_b[:,0])
    u = np.minimum(hr_a[:,1], hr_b[:,1])
    if np.any(u-l < 0): return None # Return None if no overlap.
    return np.array([l, u]).T

def hr_mbb(hr_a, hr_b):
    """
    Find minimum axis-aligned bounding box of two hyperrectangles.
    """
    return np.array([np.minimum(hr_a[:,0], hr_b[:,0]), np.maximum(hr_a[:,1], hr_b[:,1])]).T

def closest_point_in_hr(x, hr):
    """
    Given a point x and hyperrectangle hr, find the point
    inside hr that is closest to x. This is the same point for all p-norms.
    """
    return np.array([hrd[0] if hrd[0] > xd else (
                     hrd[1] if hrd[1] < xd else (
                     xd)) for xd, hrd in zip(x, hr)])

def project(nodes, dims, maximise=False, resolution=None):
    """
    Project a list of nodes onto dims and list all the hyperrectangular intersections.
    """
    dims = nodes[0].space.idxify(dims)
    # List all the unique thresholds along each dim.
    thresholds = [{} for _ in dims]    
    for node in nodes:
        for i, d in enumerate(dims): 
            for t, open_or_close in zip(node.hr_max[d] if maximise else node.hr_min[d], (0,1)):
                # Store whether the node is "opened" or "closed" at this threshold.
                if t not in thresholds[i]: thresholds[i][t] = [set(), set()]
                thresholds[i][t][open_or_close].add(node)
    # Sort the thresholds along each dim.
    thresholds = [sorted([(k,v) for k,v in thresholds[i].items()], key=lambda x: x[0]) for i in range(len(dims))]      
    # If resolution specified, reduce the number of thresholds accordingly.
    if resolution is not None:
        for i, t in enumerate(thresholds):
            t_filtered, idx_last, t_last, unallocated_close = [], -1, -np.inf, set()
            for idx in range(len(t)):
                t_idx, (new_open, new_close) = t[idx]
                if t_idx - t_last >= resolution[i]:
                    # If resolution test passed, keep this threshold.
                    t_filtered.append((t_idx, [new_open, unallocated_close | new_close]))
                    idx_last += 1; t_last = t_idx # This becomes the new last threshold.
                    unallocated_close = set() # All allocated.
                else:
                    # Otherwise, discard and reallocate nodes.
                    t_filtered[idx_last][1][0].update(new_open) # Open to last.
                    unallocated_close.update(new_close) # Closed to next (which we've not found yet!)                
            if unallocated_close != set():
                # If any unallocated remaining, add the final threshold (will violate resolution).
                t_filtered.append((t_idx, [set(), unallocated_close]))
            thresholds[i] = t_filtered
            print(i, len(t),'->',len(t_filtered))
    # Iterate through all the hyperrectangular tiling induced by the thresholds,
    # keeping track of the "open" nodes along each dimension.
    open_nodes, projections = [set() for _ in dims], []    
    for indices in product(*[range(len(t)) for t in thresholds]):
        hr = []
        for i, (idx, t) in enumerate(zip(indices, thresholds)):
            # Update the set of open nodes along this dim.
            new_open, new_close = t[idx][1]
            open_nodes[i] = (open_nodes[i] | new_open) - new_close
            # Limits of hyperrectangle along this dim.
            try: hr.append([t[idx][0],t[idx+1][0]])
            except: hr = None; break # This is triggered when idx is the max for this dim.
        if hr is not None:
            hr = np.array(hr)
            # The overlapping nodes are those that are open along all dims.
            overlapping_nodes = set.intersection(*open_nodes)
            # Only store if there are a nonzero number of overlapping nodes.
            if len(overlapping_nodes) > 0: projections.append([hr, overlapping_nodes])
    # print('Projection complete')
    return projections
    
# ===============================
# OTHER

def round_sf(X, sf):
    """
    Round a float to the given number of significant figures.
    """
    if sf is None: return X
    try: 
        # For single value.
        return np.format_float_positional(X, precision=sf, unique=False, fractional=False, trim='k') 
    except: 
        # For iterable.
        return f"[{','.join(round_sf(x, sf) for x in X)}]" 

def gather(nodes, *attributes, transpose=False):
    """
    Gather attributes from each node in the provided list.
    """
    results = []
    for attr in attributes:
        if attr is None: results.append([None])
        else: results.append([node[attr] for node in nodes])
    if len(results) == 1: return results[0]
    if transpose: return list(zip(*results))
    return results

def weighted_average(nodes, dims, hr=None, intersect_dims=None):
    """
    Average of means from several nodes along dims, weighted by population.
    If a hr is specified, additionally weight by overlap ratio along dims (using node.hr_min).
    NOTE: This encodes a strong assumption of uniform data distribution within node.hr_min.
    """
    nodes = list(nodes) # Need to have ordering.
    if len(nodes) == 1: return nodes[0].mean[dims]
    w = np.array([node.num_samples for node in nodes])
    if hr is not None:
        zero_hr_width = (hr[:,1] - hr[:,0]) == 0
        r = []
        for node in nodes:
            node_hr = node.hr_min[intersect_dims] # NOTE: Always uses hr_min, not hr_max.
            inte = hr_intersect(node_hr, hr)
            node_hr_width = node_hr[:,1] - node_hr[:,0]
            node_hr_width_corr = node_hr_width.copy()
            node_hr_width_corr[node_hr_width==0] = 1 # Prevent div/0 error.
            ratios = (inte[:,1] - inte[:,0]) / node_hr_width_corr
            ratios[node_hr_width==0] = 1 # Prevent zero ratio in same circumstance.
            ratios[zero_hr_width] = 1 # Prevent zero ratio if hr is conditioned.
            r.append(np.prod(ratios))
        w = w * r
    return np.average([node.mean[dims] for node in nodes], axis=0, weights=w)
