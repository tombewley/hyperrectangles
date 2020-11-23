import numpy as np

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
    to a list with None values as placeholders, which can be used in calculations.
    """
    dim_list = [None for _ in dim_names]
    for dim, value in dim_dict.items():
        dim_list[dim_names.index(dim)] = value
    return dim_list

def round_sf(X, sf):
    def _r(x): return np.format_float_positional(x, precision=sf, unique=False, fractional=False, trim='k')
    try: return _r(X) # For single value.
    except: return f"({', '.join(_r(x) for x in X)})" # For iterable.
