from .utils import *
import numpy as np
import matplotlib as mpl
import networkx as nx
from tqdm import tqdm

def show_samples(node, vis_dims, colour_dim=None, alpha=None, spark=False, ax=None):
    """
    Scatter plot across vis_dims of all the samples contained in node.
    """
    assert len(vis_dims) == 2
    # Allow dim_names to be specified instead of numbers.
    if type(vis_dims[0]) == str: vis_dims = [node.source.dim_names.index(v) for v in vis_dims]
    # if type(colour_dim) == str: colour_dim = node.source.dim_names.index(colour_dim)
    x = node.source.data[node.sorted_indices[:,0][:,None], vis_dims]
    lims = [[x[:,0].min(), x[:,0].max()], [x[:,1].min(), x[:,1].max()]]
    if ax is None: 
        if spark: ax = _ax_spark(ax, lims)
        else: _, ax = mpl.pyplot.subplots()#figsize=(8,8))
    # Automatically calculate alpha.
    if alpha is None: alpha = 1 / len(x)**0.25
    if spark:
        ax.scatter(x[:,0], x[:,1], s=0.5, c='#fe5d02', alpha=alpha) 
        y = node.mean[vis_dims[1]]
        ax.plot(lims[0], [y,y], c='k')
    else:
        ax.scatter(x[:,0], x[:,1], s=0.5, c='k', alpha=alpha) 
        ax.set_xlabel(node.source.dim_names[vis_dims[0]])
        ax.set_ylabel(node.source.dim_names[vis_dims[1]])
        
# ==========================================
# Functions below this line use whole trees.

def show_lines(tree, attributes, max_depth=np.inf, maximise=False, show_spread=False, ax=None):
    """
    Given a tree with one split_dim, display one or more attributes using horizontal lines for each node.
    TODO: Adapt for slicing.
    """
    assert len(tree.split_dims) == 1
    num_attributes = len(attributes)
    if show_spread: 
        attributes += [(('std',a[1]) if a[0]=='mean' else 
                       (('q1q3',a[1]) if a[0]=='median' else None)) 
                       for a in attributes]
    # Collect the list of nodes to show.
    nodes_to_show = tree.filter(max_depth=np.inf)
    values = _collect_attributes(nodes_to_show, attributes)
    # Create new axes if needed.
    if ax is None: _, ax = mpl.pyplot.subplots()#figsize=(9,8))
    split_dim_name = tree.root.source.dim_names[tree.split_dims[0]]
    ax.set_xlabel(split_dim_name)
    # Colour cycle.
    colours = mpl.pyplot.rcParams['axes.prop_cycle'].by_key()['color']
    for i, attr in enumerate(attributes[:num_attributes]):
        for n, (node, value) in enumerate(zip(nodes_to_show, values[i])):
            mn, mx = node.bb_max[tree.split_dims[0]] if maximise else node.bb_min[tree.split_dims[0]]
            mn = max(mn, tree.root.bb_min[tree.split_dims[0],0])
            mx = min(mx, tree.root.bb_min[tree.split_dims[0],1])
            ax.plot([mn,mx], [value, value], c=colours[i], label=(f'{attr[0]} of {attr[1]}' if n == 0 else None))
            if maximise and n > 0: ax.plot([mn,mn], [value_last, value], c=colours[i]) # Vertical connecting lines.
            if show_spread: # Visualise spread using a rectangle behind.
                if attr[0] == 'mean': # Use standard deviation.
                    std = values[i+num_attributes][n]
                    spread_mn, spread_mx = value - std, value + std      
                elif attr[0] == 'median': # Use quartiles.
                    spread_mn, spread_mx = values[i+num_attributes][n]
                ax.add_patch(_make_rectangle([[mn,mx],[spread_mn,spread_mx]], colours[i], edge_colour=None, alpha=0.25))
            value_last = value
    ax.legend()
    return ax

def show_rectangles(tree, vis_dims=None, attribute=None, slice_dict={}, interval_dict={},
                    max_depth=np.inf, maximise=False, cmap_lims=None, fill_colour='w', edge_colour=None, ax=None):
    """
    Given a tree with two split_dims, display one attribute using a coloured rectangle for each node.
    """
    if vis_dims is None:
        assert len(tree.split_dims) == 2
        vis_dims = tree.split_dims
    else:
        assert len(slice_dict) == len(tree.split_dims) - len(vis_dims)
        # Allow dim_names to be specified instead of numbers.
        if type(vis_dims[0]) == str: vis_dims = [tree.root.source.dim_names.index(v) for v in vis_dims] 
    # Set up axes.
    ax = _ax_setup(ax, tree, vis_dims, attribute=attribute, slice_dict=slice_dict, interval_dict=interval_dict)
    # Collect the list of nodes to show.
    nodes_to_show = tree.filter(slice_dict=slice_dict, maximise_to_slice=maximise, interval_dict=interval_dict, max_depth=max_depth)
    values = _collect_attributes(nodes_to_show, [attribute])
    # Extract bounding boxes.
    bbs = [(_bb_clip(node.bb_max[vis_dims], tree.root.bb_min[vis_dims]) if maximise 
            else node.bb_min[vis_dims]) for node in nodes_to_show]
    if attribute is not None:
        # Fill according to attribute value.
        _lims_and_values_to_rectangles(ax, bbs, values[0], cmap=_cmap(attribute), cmap_lims=cmap_lims, edge_colour=edge_colour)
    else: 
        # White fill (adds black borders if none specified.)
        _lims_and_values_to_rectangles(ax, bbs, fill_colour=fill_colour, edge_colour=edge_colour)
    return ax

def show_difference_rectangles(tree_a, tree_b, attribute, max_depth=np.inf, maximise=False, cmap_lims=None, edge_colour=None, ax=None):
    """
    Given two trees with the same two split_dims, display rectangles coloured by the differences in the given attribute.
    TODO: Adapt for slicing. 
    """
    raise NotImplementedError("Need to clip intersection to *inner* of tree_a,root.bb_min and tree_b,root.bb_min")
    assert len(tree_a.split_dims) == 2 and tree_a.split_dims == tree_b.split_dims
    # Set up axes.
    ax = _ax_setup(ax, tree_a, tree_a.split_dims, attribute=attribute, diff=True, tree_b=tree_b)    
    # Collect the lists of nodes to show.
    nodes_a = tree_a.filter(max_depth=max_depth)
    values_a = _collect_attributes(nodes_a, [attribute])
    nodes_b = tree_b.filter(max_depth=max_depth)
    values_b = _collect_attributes(nodes_b, [attribute])
    # Compute the pairwise intersections between nodes.
    intersections = []; diffs = []
    for node_a, value_a in zip(nodes_a, values_a[0]):
        for node_b, value_b in zip(nodes_b, values_b[0]):
            inte = _get_intersection(node_a, node_b, tree_a.split_dims, maximise)
            if inte is not None: # Only store if intersection is non-empty.
                intersections.append(inte)
                diffs.append(value_a - value_b)
    # Create rectangles.
    _lims_and_values_to_rectangles(ax, intersections, diffs, cmap=_cmap(attribute), cmap_lims=cmap_lims, edge_colour=edge_colour)
    return ax

def show_derivatives(tree, max_depth=np.inf, scale=1, pivot='tail', ax=None):
    """
    Given a tree with two split dimensions, show the derivatives across those same dimensions.
    TODO: Adapt for slicing.
    """
    assert len(tree.split_dims) == 2
    # Set up axes.
    ax = _ax_setup(ax, tree, tree.split_dims, derivs=True)    
    # Collect the mean locations and derivative values.
    attributes = [('mean',s) for s in vis_dim_names] + [('mean',f'd_{s}') for s in vis_dim_names]
    values = _collect_attributes(tree.filter(max_depth=max_depth), attributes)
    # Create arrows centred at means.    
    mpl.pyplot.quiver(values[0], values[1], values[2], values[3], 
                      pivot=pivot, angles='xy', scale_units='xy', units='inches', 
                      color='k', scale=1/scale, width=0.02, minshaft=1)
    return ax

def show_transition_graph(tree, layout_dims=None, highlight_path=None, alpha=False, ax=None):
    """
    xxx
    """
    assert tree.transition_graph is not None
    G = tree.transition_graph
    G.add_edge("T", "I", alpha=0)
    if layout_dims is not None:
        assert len(layout_dims) == 2 
        # Allow dim_names to be specified instead of numbers.
        if type(layout_dims[0]) == str: layout_dims = [tree.root.source.dim_names.index(v) for v in layout_dims] 
        if ax is None: ax = _ax_setup(ax, tree, layout_dims)
        pos = {}; # fixed = []
        for node in G.nodes():
            if node not in ("I","T"): # Not initial/terminal.
                # Position node at the mean of its constituent samples.
                pos[node] = node.mean[layout_dims]
                # fixed.append(node)
        pos["I"] = np.array([0, 0.45])
        pos["T"] = np.array([0, 0.2])
    else:
        # If no layout_dims, arrange using spring forces.
        pos = nx.spring_layout(G)
        if ax is None: _, ax = mpl.pyplot.subplots()#figsize=(12,12))    
    # Draw nodes and labels.
    nx.draw_networkx_nodes(G, pos=pos, node_color=["#a8caff" for _ in range(len(tree.leaves))] + ["#74ad83","#e37b40"])
    nx.draw_networkx_labels(G, pos=pos, labels=nx.get_node_attributes(G, "idx"))
    # If highlight_path specified, highlight it in a different colour.
    if highlight_path is not None:
        h = set((highlight_path[i], highlight_path[i+1]) for i in range(len(highlight_path)-1))
        edge_colours = []
        for edge in G.edges:
            if edge in h: edge_colours.append("r")
            else: edge_colours.append("k")
    else: edge_colours = "k"
    arcs = nx.draw_networkx_edges(G, pos=pos, connectionstyle="arc3,rad=0.2", edge_color=edge_colours)
    # Set alpha individually for each non-highlighted edge.
    if alpha:
        for arc, (_,_,attr), c in zip(arcs, G.edges(data=True), edge_colours): 
            if c != "r": arc.set_alpha(attr["alpha"])
    # Retrieve axis ticks which networkx likes to hide.
    if layout_dims is not None: 
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    return ax

# ===========================
# SHAP-based visualisations.

def show_shap_dependence(tree, node, shap_dims, vis_dim=None, colour_dim=None, subsample=None):
    """
    Using all the samples at a node, build a SHAP dependence plot for shap_dims[0] w.r.t. shap_dims[1].
    Scatter points along shap_dims[1] *or* a specified vis_dim, and optionally colour points by colour_dim.
    TODO: Remove interaction effects with "deinteraction_dim"
    """
    # Allow dim_names to be specified instead of numbers.
    if vis_dim is None: vis_dim = shap_dims[1]
    if type(vis_dim) == str: vis_dim = tree.root.source.dim_names.index(vis_dim)
    if type(shap_dims[0]) == str: shap_dims = [tree.root.source.dim_names.index(s) for s in shap_dims]
    if type(colour_dim) == str: colour_dim = tree.root.source.dim_names.index(colour_dim)
    d, c = [], []
    for idx in tqdm(subsample_sorted_indices(node.sorted_indices, subsample)[:,0]):
        # Compute SHAP value for each sample.
        x = node.source.data[idx]
        s = tree.shap(x, shap_dims[0])
        d.append([x[vis_dim], s[shap_dims[1]]])
        if colour_dim is not None: c.append(x[colour_dim])
    d, c = np.array(d), np.array(c) 
    # Set up figure.
    _, ax = mpl.pyplot.subplots()#figsize=(8,4))
    ax.set_xlabel(tree.root.source.dim_names[vis_dim])    
    ax.set_ylabel(f'SHAP for {tree.root.source.dim_names[shap_dims[0]]} w.r.t. {tree.root.source.dim_names[shap_dims[1]]}')
    if colour_dim is None: colours = 'k'
    else:
        cmap = (mpl.cm.copper,'copper')
        mn, mx = np.min(c), np.max(c)
        if mx == mn: fill_colours = [cmap[0](.5) for _ in c]
        else: colours = [cmap[0](v) for v in (c - mn) / (mx - mn)]
        # Create colour bar.
        norm = mpl.colors.Normalize(vmin=mn, vmax=mx)
        cbar = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap[1]), ax=ax)
        cbar.set_label(tree.root.source.dim_names[colour_dim], rotation=270)
    ax.scatter(d[:,0], d[:,1], s=2, c=colours)
    return ax

# ========================================================

def _collect_attributes(nodes, attributes):
    """
    Collect a set of attributes from each node in the provided list.
    """
    values = []
    for attr in attributes:
        if attr is None: values.append(None)
        else:
            # Allow dim_name to be specified instead of number.
            if type(attr[1]) == str: dim = nodes[0].source.dim_names.index(attr[1])
            if len(attr) == 3 and type(attr[2]) == str: dim2 = nodes[0].source.dim_names.index(attr[2])
            # Mean, standard deviation, or sqrt of covarance (std_c).
            if attr[0] == 'mean':
                values.append(np.array([node.mean[dim] for node in nodes]))
            elif attr[0] == 'std':
                values.append(np.sqrt(np.array([node.cov[dim,dim] for node in nodes])))
            elif attr[0] == 'std_c':
                values.append(np.sqrt(np.array([node.cov[dim,dim2] for node in nodes])))
            elif attr[0] in ('median','iqr','q1q3'):
                # Median, interquartile range, or lower and upper quartiles.
                v = []
                for node in nodes:
                    q1, q2, q3 = np.quantile(node.source.data[node.sorted_indices[:,dim],dim], (.25,.5,.75))
                    if attr[0] == 'median': v.append(q2)
                    elif attr[0] == 'iqr': v.append(q3-q1)
                    elif attr[0] == 'q1q3': v.append((q1,q3))
                values.append(v)
    return values

def _ax_setup(ax, tree, vis_dims, attribute=None, diff=False, tree_b=None, derivs=False, slice_dict={}, interval_dict={}):
    if ax is None: _, ax = mpl.pyplot.subplots()#figsize=(9,8))
    vis_dim_names = [tree.root.source.dim_names[v] for v in vis_dims]
    ax.set_xlabel(vis_dim_names[0]); ax.set_ylabel(vis_dim_names[1])
    title = tree.name
    if diff: title += f' vs {tree_b.name}\n$\Delta$ in {attribute[0]} of {attribute[1]}'
    elif attribute: title += f'\n{attribute[0]} of {attribute[1]}'
    elif derivs: title += '\nTime derivatives'
    if slice_dict != {}: title += '\nSlice at '+', '.join([f'{d} = {v}' for d, v in slice_dict.items()])
    if interval_dict != {}: title += '\nFilter at '+', '.join([f'{d} $\in$ {v}' for d, v in interval_dict.items()])
    ax.set_title(title)
    return ax

def _ax_spark(ax, lims):
    if ax is None: _, ax = mpl.pyplot.subplots(figsize=(1.2,1.3))
    ax.tick_params(axis='both', bottom=True, left=True, top=False, right=False, labelsize=8, pad=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.75)
    ax.spines['bottom'].set_linewidth(0.75)
    ax.figure.autofmt_xdate(ha='center') #, rotation=90)
    ax.set_xticks(lims[0])
    ax.set_yticks(lims[1])
    return ax

def _get_intersection(node_a, node_b, dims, maximise):
    """
    Find intersection between either the maximal or minimal bounding boxes for two nodes.
    """
    bb_a, bb_b = (node_a.bb_max[dims], node_b.bb_max[dims]) if maximise else (node_a.bb_min[dims], node_b.bb_min[dims])
    l = np.maximum(bb_a[:,0], bb_b[:,0])
    u = np.minimum(bb_a[:,1], bb_b[:,1]) 
    if np.any(u-l <= 0): return None # Return None if no overlap.
    return np.array([l, u]).T

def _bb_clip(bb, clip):
    bb[:,0] = np.maximum(bb[:,0], clip[:,0])
    bb[:,1] = np.minimum(bb[:,1], clip[:,1])
    return bb

def _lims_and_values_to_rectangles(ax, lims_list, values=None, cmap=None, cmap_lims=None, fill_colour='w', edge_colour=None):
    """xxx"""
    if values is not None:
        # Compute fill colour.
        if cmap_lims is None: mn, mx = np.min(values), np.max(values)
        else: mn, mx = cmap_lims
        if mx == mn: fill_colours = [cmap[0](.5) for _ in values]
        else: fill_colours = [cmap[0](v) for v in (values - mn) / (mx - mn)]
        # Create colour bar.
        norm = mpl.colors.Normalize(vmin=mn, vmax=mx)
        ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap[1]), ax=ax)
    else:
        fill_colours = [fill_colour for _ in lims_list]
        if fill_colour == 'w' and edge_colour == None: edge_colour = 'k' # Show lines by default if white fill.
    for lims, fill_colour in zip(lims_list, fill_colours):
        ax.add_patch(_make_rectangle(lims, fill_colour, edge_colour, alpha=1))    
    ax.relim(); ax.autoscale_view()

def _make_rectangle(lims, fill_colour, edge_colour, alpha):
    (xl, xu), (yl, yu) = lims
    return mpl.patches.Rectangle(xy=[xl,yl], width=xu-xl, height=yu-yl, facecolor=fill_colour, alpha=alpha, edgecolor=edge_colour, lw=0.5, zorder=-1) 

def _cmap(attribute):
    if attribute[0] in ('std','std_c','iqr'): return (mpl.cm.coolwarm, 'coolwarm') # Reverse for measures of spread.
    else:                                     return (mpl.cm.coolwarm_r, 'coolwarm_r')                 