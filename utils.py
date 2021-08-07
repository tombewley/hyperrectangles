import numpy as np
import bisect
from itertools import product

# ===============================
# OPERATIONS USED FOR VARIANCE-BASED SPLITTING

def increment_mean_and_var_sum(n, mean, var_sum, x, sign):
    """
    Welford's online algorithm for incremental sum-of-variance computation,
    adapted from https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
    """
    d_last = x - mean
    mean = mean + (sign * (d_last / n)) # Can't do += because this modifies the NumPy array in place!
    d = x - mean
    var_sum = var_sum + (sign * (d_last * d))
    return mean, np.maximum(var_sum, 0) # Clip at zero.

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
    print('Projection complete')
    return projections
    
# ===============================
# OTHER

def round_sf(X, sf):
    """
    Round a float to the given number of significant figures.
    """
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
            node_bb = node.bb_min[intersect_dims] # NOTE: Always uses bb_min, not bb_max
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