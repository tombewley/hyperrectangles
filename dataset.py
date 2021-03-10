import numpy as np

class Dataset:
    """
    Class for a dataset, which lives inside a space. From the outside it appears similar to a NumPy array, but has
    additional functionality for named indexing, filtering and maintained sorting.
    """
    def __init__(self, space, numpy, sort=None): 
        self.space = space
        self.numpy = np.array(numpy)
        if sort is None: sort = np.argsort(self.numpy, axis=0)
        self.sort = sort
        
    def __getitem__(self, key):
        """
        Retrieve data in more ways than allowed by NumPy.
        """
        if type(key) == dict: key = self.space.listify(key, duplicate_singletons=True)
        if type(key) == list: return self.bb_filter(key)
        if type(key) == set: return self.numpy[list(key)] # Unordered.
        return self.numpy[key]

    def bb_filter(self, bb):
        """
        Making use of the split_sorted_indices function, filter sorted_indices using a bb.
        Allow bb to be specified as a dict.
        """
        sort = self.sort
        for split_dim, lims in enumerate(bb):
            if lims is None: continue # If nothing specified for this lim.
            for lu, lim in enumerate(lims):
                if np.isfinite(lim):
                    X = self.numpy[sort[:,split_dim], split_dim] # Must reselect each time.
                    if lu == 0:
                        # For lower limit, bisect to the right.
                        split_index = bisect.bisect_right(X, lim)
                        _, sort = split_sorted_indices(sort, split_dim, split_index)
                    else:
                        # For upper limit, bisect to the left.
                        split_index = bisect.bisect_left(X, lim)
                        sort, _ = split_sorted_indices(sort, split_dim, split_index)    
        return sort # Subset(self, sort)

# class Subset(Dataset):
#     def __init__(self, ): return