


class DFSM:
    """
    Class for a dynamic FSM, ...
    """


def segments_to_datasets(self):
    """
    Can use to grow another tree for each leaf.
    """
    X = []
    for l in self.model.leaves:
        X.append(np.array([[s["step"], self.leaves_plus.index(s["next"])] 
                            for s in _seg_filter(self.segments, leaves=[l])]))
    return X     

def split(self, names, by="ep", at="max_divergence"):
    """
    Split the data for this FSM along the given dimension, yielding multiple child FSMs.
    """
    if type(at) in (float, int): at = (at,)
    if type(at) == tuple:
        assert len(names) == len(at) + 1
        segment_splits = {name:[] for name in names}
        i = 0
        for s in self.segments:
            if i < len(at) and s[by] >= at[i]: i += 1
            segment_splits[names[i]].append(s)
    return (FSM(name, self.model, segments=seg) for name, seg in segment_splits.items())
