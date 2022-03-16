import numpy as np
import bisect
from itertools import product
import numba

# ===============================
# OPERATIONS USED FOR VARIANCE-BASED SPLITTING

def variance_based_split_finder(node, split_dims, eval_dims, min_samples_leaf, store_all_qual=False):
    """
    Try splitting the node along several split_dims, measuring quality using eval_dims.
    Return the best split from each dim.
    """
    parent_mean = node.mean[eval_dims]
    parent_var_sum = node.var_sum[eval_dims]
    parent_num_samples = node.num_samples
    var_scale = node.space.global_var_scale[eval_dims]
    splits, greedy_gains = [], []
    if store_all_qual: node.all_split_thresholds, node.all_qual = {}, {}
    for split_dim in split_dims: # TODO: May be possible to realise further speed improvements by parallelising with numba.prange
        split_data = node.space.data[node.sorted_indices[:,split_dim][:,None],split_dim]
        eval_data = node.space.data[node.sorted_indices[:,split_dim][:,None],eval_dims]
        valid_split_indices, qual, split_index, qual_max = qual_weighted_var_sum(split_data, eval_data, min_samples_leaf,
                                                           parent_mean, parent_var_sum, parent_num_samples, var_scale)
        if valid_split_indices is not None:
            # greedy_gains.append(gains_this_dim[greedy]) # NOTE: Not implemented
            # Store split info.
            splits.append((split_dim, split_index, qual_max))
            # If applicable, store all split thresholds and quality values
            if store_all_qual:
                node.all_split_thresholds[split_dim] = (split_data[valid_split_indices-1] + split_data[valid_split_indices]) / 2
                node.all_qual[split_dim] = qual
    return splits, np.array(greedy_gains)

@numba.njit(cache=True, parallel=False)
def qual_weighted_var_sum(split_data, eval_data, min_samples_leaf, parent_mean, parent_var_sum, parent_num_samples, var_scale):
    """
    Identify and evaluate all valid splits of node along split_dim, incrementally calculating variance sums along eval_dims.
    Calculate split quality = sum of reduction in dimension-scaled variance sums and find the greedy split.
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

    # Apply two kinds of constraint to the split point:
    #   (1) Must be a "threshold" point where the samples either side do not have equal values
    valid_split_indices = np.where(split_data[1:] - split_data[:-1])[0] + 1 # NOTE: 0 will not be included
    #   (2) Must obey min_samples_leaf
    mask = np.logical_and(valid_split_indices >= min_samples_leaf, valid_split_indices <= parent_num_samples-min_samples_leaf)
    valid_split_indices = valid_split_indices[mask]
    # Cannot split on a dim if there are no valid split points
    if len(valid_split_indices) == 0: return None, None, None, None
    # greedy_gains.append(np.full(len(eval_dims), np.nan)) # NOTE: Not implemented
    max_num_left = valid_split_indices[-1] + 1 # +1 needed
    mean = np.zeros((2, max_num_left, eval_data.shape[1]))
    var_sum = mean.copy()
    mean[1,0] = parent_mean
    var_sum[1,0] = parent_var_sum
    for num_left in range(1, max_num_left): # Need to start at 1 for incremental calculation to work
        num_right = parent_num_samples - num_left
        x = eval_data[num_left-1]
        mean[0,num_left], var_sum[0,num_left] = increment_mean_and_var_sum(num_left,  mean[0,num_left-1], var_sum[0,num_left-1], x, 1)
        mean[1,num_left], var_sum[1,num_left] = increment_mean_and_var_sum(num_right, mean[1,num_left-1], var_sum[1,num_left-1], x, -1)                    
    gains = var_sum[1,0] - var_sum[:,valid_split_indices,:].sum(axis=0)
    qual = (gains * var_scale).sum(axis=1)
    # Greedy split is the one with the highest quality
    greedy = np.argmax(qual)
    split_index = valid_split_indices[greedy]
    qual_max = qual[greedy]
    return valid_split_indices, qual, split_index, qual_max

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

def bb_filter_sorted_indices(space, sorted_indices, bb):
    """
    Making use of the split_sorted_indices function, filter sorted_indices using a bb.
    Allow bb to be specified as a dict.
    """
    for split_dim, lims in enumerate(space.listify(bb)):
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

def bb_intersect(bb_a, bb_b):
    """
    Find intersection between two bounding boxes.
    """
    l = np.maximum(bb_a[:,0], bb_b[:,0])
    u = np.minimum(bb_a[:,1], bb_b[:,1]) 
    if np.any(u-l < 0): return None # Return None if no overlap.
    return np.array([l, u]).T

def bb_clip(bb, clip):
    """
    Clip a bounding box using another.
    """
    bb[:,0] = np.clip(bb[:,0], clip[:,0], clip[:,1])
    bb[:,1] = np.clip(bb[:,1], clip[:,0], clip[:,1])
    return bb

def closest_point_in_bb(x, bb):
    """
    Given a point x and hyperectangular bounding box bb, find the point
    inside bb that is closest to x. This is the same point for all p-norms.
    """
    return np.array([bbd[0] if bbd[0] > xd else (
                     bbd[1] if bbd[1] < xd else (
                     xd)) for xd, bbd in zip(x, bb)]) 

def project(nodes, dims, maximise=False, resolution=None):
    """
    Project a list of nodes onto dims and list all the regions of intersection.
    """
    dims = nodes[0].space.idxify(dims)
    # List all the unique thresholds along each dim.
    thresholds = [{} for _ in dims]    
    for node in nodes:
        for i, d in enumerate(dims): 
            for t, open_or_close in zip(node.bb_max[d] if maximise else node.bb_min[d], (0,1)):
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
    # Iterate through all Cartesian products of intervals (bounding boxes), 
    # keeping track of the "open" nodes in each bounding box.
    open_nodes, projections = [set() for _ in dims], []    
    for indices in product(*[range(len(t)) for t in thresholds]):
        bb = []
        for i, (idx, t) in enumerate(zip(indices, thresholds)):
            # Update the set of open nodes along this dim.
            new_open, new_close = t[idx][1]
            open_nodes[i] = (open_nodes[i] | new_open) - new_close
            # Limits of bounding box along this dim.
            try: bb.append([t[idx][0],t[idx+1][0]]) 
            except: bb = None; break # This is triggered when idx is the max for this dim.
        if bb is not None:
            bb = np.array(bb)
            # The overlapping nodes are those that are open along all dims.
            overlapping_nodes = set.intersection(*open_nodes)
            # Only store if there are a nonzero number of overlapping nodes.
            if len(overlapping_nodes) > 0: projections.append([bb, overlapping_nodes])
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

def weighted_average(nodes, dims, bb=None, intersect_dims=None):
    """
    Average of means from several nodes along dims, weighted by population.
    If a bb is specified, additionally weight by overlap ratio along dims (using node.bb_min).
    NOTE: This encodes a strong assumption of uniform data distribution within node.bb_min.
    """
    nodes = list(nodes) # Need to have ordering.
    if len(nodes) == 1: return nodes[0].mean[dims]
    w = np.array([node.num_samples for node in nodes])
    if bb is not None:
        zero_bb_width = (bb[:,1] - bb[:,0]) == 0
        r = []
        for node in nodes:
            node_bb = node.bb_min[intersect_dims] # NOTE: Always uses bb_min, not bb_max.
            inte = bb_intersect(node_bb, bb)
            node_bb_width = node_bb[:,1] - node_bb[:,0]
            node_bb_width_corr = node_bb_width.copy()
            node_bb_width_corr[node_bb_width==0] = 1 # Prevent div/0 error.
            ratios = (inte[:,1] - inte[:,0]) / node_bb_width_corr
            ratios[node_bb_width==0] = 1 # Prevent zero ratio in same circumstance.
            ratios[zero_bb_width] = 1 # Prevent zero ratio if bb is conditioned.
            r.append(np.prod(ratios))
        w = w * r
    return np.average([node.mean[dims] for node in nodes], axis=0, weights=w)