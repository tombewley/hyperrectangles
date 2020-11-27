import numpy as np
from itertools import product

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

def increment_mean_and_cov_sum(n, mean, cov_sum, x, sign):
    """
    Adaptation of the above to work with the whole covariance matrix.
    """
    d_last = x - mean
    mean = mean + (sign * (d_last / n))
    d = x - mean
    cov_sum = cov_sum + (sign * np.outer(d_last, d))
    return mean, cov_sum

def cov_to_r2(cov):
    """
    Convert a covariance matrix to R^2 matrix, keeping only elements above the diagonal.
    Adapted from https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b.
    """
    var = np.maximum(np.diag(cov), 0)
    r2 = (cov**2) / np.outer(var, var)
    r2[cov == 0] = 0
    return np.triu(r2, k=1)

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

def dim_dict_to_list(dim_dict, dim_names):
    """
    Convert a convenient dictionary representation of per-dimension information
    to a list with Nones as placeholders, which can be used in calculations.
    """
    dim_list = [None for _ in dim_names]
    for dim, value in dim_dict.items():
        dim_list[dim_names.index(dim)] = value
    return dim_list

def round_sf(X, sf):
    """
    Round a float to the given number of significant figures.
    """
    def _r(x): return np.format_float_positional(x, precision=sf, unique=False, fractional=False, trim='k')
    try: return _r(X) # For single value.
    except: return f"({', '.join(_r(x) for x in X)})" # For iterable.

def gather_attributes(nodes, attributes):
    """
    Gather a set of attributes from each node in the provided list.
    """
    results = []
    for attr in attributes:
        if attr is None: results.append(None)
        else:
            # Allow dim_name to be specified instead of number.
            if type(attr[1]) == str: dim = nodes[0].source.dim_names.index(attr[1])
            if len(attr) == 3 and type(attr[2]) == str: dim2 = nodes[0].source.dim_names.index(attr[2])
            results.append(np.array([node.attr(attr) for node in nodes]))
    return results

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
    bb[:,0] = np.maximum(bb[:,0], clip[:,0])
    bb[:,1] = np.minimum(bb[:,1], clip[:,1])
    return bb

def project(nodes, dims, maximise=False, resolution=None):
    """
    Project a list of nodes onto dims and list all the regions of intersection.
    """
    if type(dims[0]) == str: dims = [nodes[0].source.dim_names.index(d) for d in dims]
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
            print(len(t),'->',len(t_filtered))
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
            if len(overlapping_nodes) > 0: projections.append((bb, overlapping_nodes))
    print('Projection complete.')
    return projections

def weighted_average(nodes, dims, bb=None, intersect_dims=None):
    """
    Average of means from several nodes along dims, weighted by population.
    If a bb is specified, additionally weight by overlap ratio along dims (using node.bb_min).
    NOTE: This encodes a strong assumption of uniform data distribution within node.bb_min.
    """
    nodes = list(nodes) # Need to have ordering.
    w = np.array([n.num_samples for n in nodes])
    if bb is not None:
        r = []
        for node in nodes:
            node_bb = node.bb_min[intersect_dims]
            node_bb_width = node_bb[:,1] - node_bb[:,0]
            node_bb_width[node_bb_width==0] = 1 # Prevent div/0 error.
            inte = bb_intersect(node_bb, bb)
            ratios = (inte[:,1] - inte[:,0]) / node_bb_width
            r.append(np.prod(ratios))
        w = w * r
    return np.average([n.mean[dims] for n in nodes], axis=0, weights=w)