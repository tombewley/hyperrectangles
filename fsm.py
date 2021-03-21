import numpy as np
import networkx as nx
from scipy.spatial.distance import jensenshannon as jsd
from tqdm import tqdm

class FSM:
    """
    Class for a finite state machine, whose states are derived from the leaves of a model
    The class has several specific methods for time series data.
    The elementary data instance for an FSM is a segment, which is a consecutive sequence of samples in the same state.
    """
    def __init__(self, name, model, segments=None, data=None, depopulate_model=False, pbar=True):
        assert {"time", "step", "ep", "return"} - set(model.space.dim_names) == set()
        self.name = name
        self.model = model.clone().depopulate() if depopulate_model else model 
        self.space = self.model.space
        self.leaves_plus = self.model.leaves + ["I", "T"]
        if segments is not None: self.segments = segments # Use precomputed segments.
        else:
            # Assemble segments from data.
            # prev_leaf = "I" corresponds to initial state. next_leaf = "T" corresponds to terminal state.
            ep_idx = self.space.dim_names.index("ep") 
            step_idx = self.space.dim_names.index("step")
            time_idx = self.space.dim_names.index("time") 
            return_idx = self.space.dim_names.index("return") 
            self.states = [State(self, l) for l in range(len(self.model.leaves))]
            m = len(self.states)
            for x in tqdm(data, desc=f"{self.name}: Building FSM from model (( {self.model} ))", disable=not pbar):
                leaves = self.model.propagate(x, mode="max") 
                assert len(leaves) == 1, "Can only compute transitions if leaves are disjoint and exhaustive."
                leaf_idx = self.model.leaves.index(next(iter(leaves)))
                time = x[time_idx]
                if time == 0 or leaf_idx != current_leaf_idx: # Start of ep or transition.
                    if time == 0: prev_leaf_idx = m
                    else: 
                        assert time == time_last + 1, "Data must be temporally ordered"
                        prev_leaf_idx = current_leaf_idx
                        current_seg["next"] = leaf_idx
                    try: self.states[current_leaf_idx].segments.append(current_seg) # Store previous segment.
                    except: pass # This happens on the very first sample.
                    current_leaf_idx = leaf_idx
                    current_seg = {
                                "prev":prev_leaf_idx, 
                                "ep":int(x[ep_idx]), 
                                "step":int(x[step_idx]), 
                                "return":x[return_idx],
                                "len":0, 
                                "next":m+1
                                }
                time_last = time; current_seg["len"] += 1
            self.states[current_leaf_idx].segments.append(current_seg) # Store very last segment.

    # Dunder/magic methods.
    def __repr__(self): return f"{self.name}: FSM from model (( {self.model} )) with {len(self)} states"    
    def __len__(self): return len(self.leaves_plus)     

    def transition_matrices(self, order=1, self_loops=False, ep=None): 
        # Collect counts from states.
        counts = np.stack(tuple(s.counts(order, self_loops, ep) for s in self.states), axis=1)
        m = len(self.states)
        if order == 1: 
            counts = np.hstack((counts, np.zeros((2,2,m+2)))) # Add empty rows for initial and terminal.
            counts_p, counts = counts # This works very nicely because of the behaviour of np.stack.
            # Copy over counts for initial and terminal from the "opposite" matrices.
            counts[m] = counts_p[:,m].copy()
            counts_p[m+1] = counts[:,m+1].copy()
            # Sanity check for 1st-order: next counts = transposed prev counts.
            if ep == None: assert (counts == counts_p.T).all()                
        else:
            # For 2nd-order, marginalise over the intermediate state.
            counts = counts.sum(axis=1)
            # Sanity check for 2nd-order: counts_2nd[i,j] <= sum_k( min(counts_1st[i,k], counts_1st[k,j]) )
            # In a Markov model with infinite samples, this would be a strict equality.
            # assert (counts <= np.minimum(self.counts[:,:,None], self.counts).sum(axis=1)).all()
        # Compute transition matrices. 
        s_n = counts.sum(axis=1, keepdims=True); s_n[s_n==0] = 1 # Prevent div/0 error.
        P = counts / s_n; P[m+1,m+1] = 1 # T is an absorbing state.
        s_p = counts.sum(axis=0, keepdims=True); s_p[s_p==0] = 1
        P_p = (counts / s_p).T; P_p[m,m] = 1 # I is an absorbing state.
        return P, P_p, counts.astype(int)

    def path(self, X, self_loops=False):
        """
        Propagate a set of samples through self.model, returning a path through the leaves.
        """
        path = []; leaf_last = None
        for x in X: 
            leaves = self.model.propagate(x, mode="max") 
            assert len(leaves) == 1
            leaf = next(iter(leaves))
            if self_loops or leaf != leaf_last: 
                path.append(leaf); leaf_last = leaf
        return path

    def eval(self, path, mode="probs", add_init_count=False):
        """
        Given a leaf path, return the list of counts, probabilities or log probabilities.
        """
        g = list(self.graph); path = [g.index(l) for l in path]
        if mode == "counts": # Counts.
            out = [self.counts[path[i],path[i+1]] for i in range(len(path)-1)]
        elif mode == "logprobs": # Log probabilities.
            out = []
            for i in range(len(path)-1): 
                if path[i] == path[i+1]: out.append(0)
                else:
                    p = self.P[path[i],path[i+1]]
                    if p <= 1e-10: out.append(-np.inf)
                    else: out.append(np.log(p))
        elif mode == "probs": # Probabilities.
            out = [1. if path[i] == path[i+1] else self.P[path[i],path[i+1]] for i in range(len(path)-1)]
        else: raise Exception()
        return ([self.counts[path[0]].sum()] if add_init_count else []) + out

    def sample(self, n=1, start="I"):
        """
        Sample a trajectory using a Markov model.
        """
        paths = []
        for _ in range(n):
            p = [start]; i = list(self.graph.nodes).index(start)
            while True:
                i = np.random.choice(range(len(self)), p=self.P[i])
                if i == len(self.model) + 1: p.append("T"); break
                p.append(self.model.leaves[i])
            paths.append(p)
        if n == 1: return paths[0]    
        return paths

    def get_steps(self, leaf=None, prev=None, nxt=None, first=False, flatten=True):
        """
        Given a filter specification, return the step numbers of either:
            first = False: all samples in matching segments.
            first = True: only the first sample in each segment.
        """
        steps = []
        for s in _seg_filter(self.segments, prevs=({prev} if prev is not None else None), 
                                            leaves=({leaf} if leaf is not None else None),
                                            nexts=({nxt} if nxt is not None else None)):
            if first: steps.append([s["step"]])
            else: steps.append(list(range(s["step"],s["step"]+s["len"])))
        if flatten: steps = [i for s in steps for i in s]
        return steps

    def ep_split_manual(self, split_thresholds): 
        for s in self.states: s.ep_split_manual(split_thresholds)

    def ep_split_jsd(self, max_depth=1, min_qual=0, plot=False):
        for s in self.states: s.ep_split_jsd(max_depth, min_qual, plot)
            
# ======================================

class State:
    def __init__(self, fsm, leaf_idx): 
        self.fsm, self.leaf_idx, self.segments, self.ep_splits = fsm, leaf_idx, [], {}    

    def ep_split_manual(self, split_thresholds):
        """
        Manually split the segments in this leaf/state along the ep dimension.
        """
        assert self.ep_splits == {}
        if type(split_thresholds) in (float, int): split_thresholds = [split_thresholds]
        t = [-np.inf] + split_thresholds + [np.inf]
        self.ep_splits = {(t[i], t[i+1]):[] for i in range(len(t)-1)}
        k = list(self.ep_splits.keys()); i = 0
        for s in self.segments:
            if i < len(self.ep_splits) and s["ep"] >= k[i][1]: i += 1 
            self.ep_splits[k[i]].append(s)

    def ep_split_jsd(self, max_depth, min_qual, plot):
        """
        Automatically split the segments for this leaf/state along the ep dimension on a *per-leaf basis*,
        in order to maximise the Jensen-Shannon distance between transition probabilities.
        NOTE: By splitting along ep rather than step we ensure the counts matrices add up properly.  
        """
        assert self.ep_splits == {}
        # ---------------------------
        if plot:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots(); ax2 = ax.twinx()
            ax.set_xlabel("Episode"); ax.set_ylabel("Next state")
        # ---------------------------
        zeros = np.zeros(len(self.fsm.leaves_plus), dtype=int)
        def _recurse(segments, interval, depth):
            if depth < max_depth:
                # Initialise transition counts.
                counts_left, counts_right = zeros.copy(), zeros.copy()
                for s in segments: counts_right[s["next"]] += 1
                qual, n = [], len(segments) # segments[0]["ep"]
                for i in range(n-1):
                    # Incrementally update transition counts.
                    nxt = segments[i]["next"]; counts_left[nxt] += 1; counts_right[nxt] -= 1
                    # Only allow splits at the ends of episodes.
                    if segments[i]["ep"] != segments[i+1]["ep"]:
                        # Split quality = sqrt of minimum total count * JSD between count vectors.
                        # This SciPy implementation normalises the vectors to sum to 1.
                        qual.append(np.sqrt(min(i+1, n-i-1)) * jsd(counts_left, counts_right))
                    else: qual.append(0)
                # Greedy split is the one with the highest quality.
                greedy = np.argmax(qual)
                if qual[greedy] > min_qual:
                    # Compute numerical threshold to split at: midpoint of segments either side.
                    split_threshold = (segments[greedy]["ep"] + segments[greedy+1]["ep"]) / 2
                    segments_left, segments_right = segments[:greedy+1], segments[greedy+1:]
                    # ---------------------------   
                    if plot:
                        ax2.plot([split_threshold,split_threshold], [0,qual[greedy]], c="k")
                        q = np.array(qual)
                        t = np.array([s["ep"] for s in segments]); t = (t[:-1] + t[1:]) / 2
                        ax2.plot(t[q>0], q[q>0], c="gray")
                        ax.scatter([s["ep"] for s in segments], [s["next"] for s in segments], c=[s["return"] for s in segments], cmap="viridis", marker="|")
                        # ax.scatter([s["ep"] for s in segments_right], [s["next"] for s in segments_right], marker="|", c="r")
                    # ---------------------------
                    _recurse(segments_left, (interval[0], split_threshold), depth+1)
                    _recurse(segments_right, (split_threshold, interval[1]), depth+1)
                    return
            # If no split is made, store this interval.
            self.ep_splits[interval] = segments
        _recurse(self.segments, (-np.inf, np.inf), 0)

    def counts(self, order=1, self_loops=True, ep=None):
        """
        Use segments to build 1st- or 2nd-order transition counts.
        """
        if ep is None: segments = self.segments
        else: segments = self.ep_splits[self._get_ep_split(ep)]
        m = len(self.fsm.states)
        if order == 2: 
            assert self_loops == False
            counts = np.zeros((m+2,m+2))
        else: counts_p = np.zeros(m+2); counts_n = counts_p.copy()
        for s in segments:
            if order == 2: counts[s["prev"],s["next"]] += 1
            else:
                counts_p[s["prev"]] += 1; counts_n[s["next"]] += 1
                if self_loops: 
                    l = s["len"]-1
                    counts_p[self.leaf_idx] += l; counts_n[self.leaf_idx] += l
        if order == 2: return counts
        else: return counts_p, counts_n

    def steps_v1(self, ep=None):
        """
        Gather step indices for inbound, self-loop and outbound transitions.
        """
        if ep is None: segments = self.segments
        else: segments = self.ep_splits[self._get_ep_split(ep)]
        m = len(self.fsm.states)
        inbound, self_loops, outbound = [[] for _ in range(m+2)], [], [[] for _ in range(m+2)]
        for s in segments:
            steps = list(range(s["step"], s["step"]+s["len"]))
            inbound[s["prev"]].append(steps[0])
            self_loops += steps[1:]
            # NOTE: if outbound to terminal, subtract 1 from the step index.
            outbound[s["next"]].append(steps[-1] if s["next"] == m+1 else steps[-1]+1)
        return inbound, self_loops, outbound

    def steps_v2(self, self_loops=True):
        """
        Store step indices and next states in a dictionary.
        """
        self_idx = self.fsm.states.index(self)
        m = len(self.fsm.states)
        steps = {}     
        for s in self.segments:
            stps = list(range(s["step"], s["step"]+s["len"]))
            if self_loops: 
                for stp in stps[1:]: steps[stp] = self_idx                
            if s["next"] == m+1: steps[stps[-1]] = s["next"]
            else: steps[stps[-1]+1] = s["next"]
        return steps

    def returns(self, mode="V", ep=None):
        """
        Gather returns from segments. If mode == "Q", organise by next leaf.
        """
        if ep is None: segments = self.segments
        else: segments = self.ep_splits[self._get_ep_split(ep)]
        if mode == "V": returns = [] 
        elif mode == "Q": returns = [[] for _ in self.fsm.leaves_plus]
        for s in segments:
            if mode == "V": returns.append([s["step"], s["return"]])
            elif mode == "Q": returns[s["next"]].append([s["step"], s["return"]])  
        if mode == "V": 
            returns = np.array(returns)     
            mean = returns[:,1].mean()
        elif mode == "Q": 
            returns = [np.array(g) for g in returns]
            mean = np.array([g[:,1].mean() if g != [] else np.nan for g in returns])
        return returns, mean

    def _get_ep_split(self, ep):
        for k in self.ep_splits.keys(): 
            if ep >= k[0] and ep < k[1]: return k
        raise ValueError("No ep_split matching given ep.")

# ======================================

class Graph:
    """Graph defined by transition counts matrix."""
    def __init__(self, nodes, counts): 
        assert nodes[-2:] == ["I","T"]
        self.space = nodes[0].space
        self.graph = nx.DiGraph()
        # Create nodes.
        self.graph.add_nodes_from([(node, 
            # For attributes, use the node's meta dictionary and add an index.
            dict([('idx',i)] + list(node.meta.items())) if i < len(nodes)-2 else {'idx':node}) 
            for i, node in enumerate(nodes)])
        # # Create edges.
        mx = counts.max() # Count for single most common transition.
        for i, node in enumerate(self.graph.nodes): 
            s = counts[i].sum(); P = counts[i] / (s if s > 0 else 1) # Prevent div/0 error.
            for j, count in enumerate(counts[i]):
                if count > 0:
                    self.graph.add_edge(node, nodes[j], count=int(count), # Get rid of NumPy datatypes to allow JSON serialisation.
                        alpha=float(count/mx), 
                        cost=float(-np.log(P[j])) # Edge cost = negative log prob.
                        )

    def dijkstra(self, source, dest=None):
        """
        Use networkx's inbuilt Dijktra algorithm to find the highest-probability paths from a source leaf.
        If a destination is specified, use that. Otherwise, find paths to all other leaves.
        """
        return nx.single_source_dijkstra(self.graph, source=source, target=dest, weight="cost")

    def json(self, *attributes, clip=None):
        """
        Create json serialisable representation of graph.
        """
        g = self.graph.copy()
        relabel = {}; reattr = {}
        for node, attr in g.nodes(data=True):
            # Replace node with its index to make serialisable. 
            relabel[node] = attr["idx"]; del attr["idx"]
            try: 
                # Collect node attributes.
                reattr[node] = {**attr, **node.json(*attributes, clip=clip)}
            except: continue # For initial/terminal.
        nx.set_node_attributes(g, reattr)
        g = nx.relabel_nodes(g, relabel)
        j = nx.readwrite.json_graph.node_link_data(g)
        j["var_names"] = self.space.dim_names
        return j

    def show(self, layout_dims=None, highlight_path=None, alpha=True, ax=None):
        """
        Visualise graph using networkx.
        """
        import matplotlib.pyplot as plt
        if layout_dims is not None:
            assert len(layout_dims) == 2 
            layout_dims = self.space.idxify(layout_dims)
            if ax is None: 
                from .visualise import _ax_setup; ax = _ax_setup(ax, self, layout_dims)
            pos = {}; # fixed = []
            for node in self.graph.nodes():
                if node not in ("I","T"): # Not initial/terminal.
                    # Position node at the mean of its constituent samples.
                    pos[node] = node.mean[layout_dims]
                    print(node)
            
            pos_array = np.vstack(tuple(pos.values()))
            pos_1_mean = np.mean(pos_array[:,1])
            pos["I"] = [np.min(pos_array[:,0]), pos_1_mean]
            pos["T"] = [np.max(pos_array[:,0]), pos_1_mean]
        else:
            # If no layout_dims, arrange using spring forces.
            pos = nx.spring_layout(self.graph)
            if ax is None: _, ax = plt.subplots()#figsize=(12,12))    
        # Draw nodes and labels.
        nx.draw_networkx_nodes(self.graph, pos=pos, node_color=["#a8caff" for _ in range(len(self.graph)-2)] + ["#74ad83","#e37b40"], ax=ax)
        nx.draw_networkx_labels(self.graph, pos=pos, labels=nx.get_node_attributes(self.graph, "idx"), ax=ax)
        # If highlight_path specified, highlight it in a different colour.
        if highlight_path is not None:
            h = set((highlight_path[i], highlight_path[i+1]) for i in range(len(highlight_path)-1))
            edge_colours = []
            for edge in self.graph.edges:
                if edge in h: edge_colours.append("r")
                else: edge_colours.append("k")
        else: edge_colours = ["k" for _ in self.graph.edges]
        
        arcs = nx.draw_networkx_edges(self.graph, pos=pos, connectionstyle="arc3,rad=0.2", edge_color=edge_colours, ax=ax)
        # Set alpha individually for each non-highlighted edge.
        if alpha:
            for arc, (_,_,attr), c in zip(arcs, self.graph.edges(data=True), edge_colours):
                if c != "r": arc.set_alpha(attr["alpha"])
        # Retrieve axis ticks which networkx likes to hide.
        if layout_dims is not None: 
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        return ax

# def _seg_filter(segments, prevs=None, eps=None, steps=None, leaves=None, nexts=None):
#     if prevs is not None: segments = [s for s in segments if s["prev"] in prevs]
#     if eps is not None: segments = [s for s in segments if s["ep"] in eps]
#     if steps is not None: segments = [s for s in segments if s["step"] in steps]
#     if leaves is not None: segments = [s for s in segments if s["leaf"] in leaves]
#     if nexts is not None: segments = [s for s in segments if s["next"] in nexts]
#     return segments