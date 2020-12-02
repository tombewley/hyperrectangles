from .utils import *
import numpy as np
import matplotlib as mpl
import mpl_toolkits.mplot3d.art3d as art3d
import networkx as nx

def show_samples(node, vis_dims, colour_dim=None, alpha=None, spark=False, subsample=None, ax=None):
    """
    Scatter plot across vis_dims of all the samples contained in node.
    TODO: colour_dim.
    """
    assert len(vis_dims) in (2,3)
    # Allow dim_names to be specified instead of numbers.
    if type(vis_dims[0]) == str: vis_dims = [node.source.dim_names.index(v) for v in vis_dims]
    # if type(colour_dim) == str: colour_dim = node.source.dim_names.index(colour_dim)
    X = node.source.data[subsample_sorted_indices(node.sorted_indices, subsample)[:,0][:,None], vis_dims]
    lims = [[X[:,0].min(), X[:,0].max()], [X[:,1].min(), X[:,1].max()]]
    if ax is None: 
        if spark: ax = _ax_spark(ax, lims)
        else: _, ax = mpl.pyplot.subplots()#figsize=(8,8))
    # Automatically calculate alpha.
    if alpha is None: alpha = 1 / len(X)**0.5
    if spark:
        ax.scatter(X[:,0], X[:,1], s=0.25, c='#fe5d02', alpha=alpha) 
        y = node.mean[vis_dims[1]]
        ax.plot(lims[0], [y,y], c='k')
    else:
        ax.set_xlabel(node.source.dim_names[vis_dims[0]])
        ax.set_ylabel(node.source.dim_names[vis_dims[1]])
        if len(vis_dims) == 3: 
            ax.scatter(X[:,0], X[:,1], X[:,2], s=0.25, c='k', alpha=alpha) 
            ax.set_zlabel(node.source.dim_names[vis_dims[2]])
        else:
            ax.scatter(X[:,0], X[:,1], s=0.25, c='k', alpha=alpha) 
        
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
    nodes = list(tree.propagate({}, mode=('max' if maximise else 'min'), max_depth=np.inf))
    values = gather_attributes(nodes, attributes)
    # Create new axes if needed.
    if ax is None: _, ax = mpl.pyplot.subplots()#figsize=(9,8))
    split_dim_name = tree.root.source.dim_names[tree.split_dims[0]]
    ax.set_xlabel(split_dim_name)
    # Colour cycle.
    colours = mpl.pyplot.rcParams['axes.prop_cycle'].by_key()['color']
    for i, attr in enumerate(attributes[:num_attributes]):
        for n, (node, value) in enumerate(zip(nodes, values[i])):
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

def show_rectangles(tree, vis_dims=None, attribute=None, 
                    slice_dict={}, max_depth=np.inf, maximise=False, project_resolution=None,
                    cmap_lims=None, fill_colour=None, edge_colour=None, ax=None):
    """
    Compute the rectangular projections of nodes from tree onto vis_dims, and colour according to attribute.
    Where multiple projections overlap, compute a marginal value using the weighted_average function from utils.
    """
    if vis_dims is None:
        assert len(tree.split_dims) == 2, 'Can only leave vis_dims unspecified if |tree.split_dims| = 2.'
        vis_dims = tree.split_dims
    else:
        # Allow dim_names to be specified instead of numbers.
        if type(vis_dims[0]) == str: vis_dims = [tree.root.source.dim_names.index(v) for v in vis_dims] 
    # Set up axes.
    ax = _ax_setup(ax, tree, vis_dims, attribute=attribute, slice_dict=slice_dict)
    # Collect the list of nodes to show.
    if slice_dict != {}: slice_list = dim_dict_to_list(slice_dict, tree.root.source.dim_names)
    else: slice_list = slice_dict
    nodes = list(tree.propagate(slice_list, mode=('max' if maximise else 'min'), max_depth=max_depth))
    if not(np.array_equal(vis_dims, tree.split_dims)) and attribute is not None:
        # If require projection.
        # TODO: Can avoid this if conditioned along (split_dims - vis_dims).
        assert attribute[0] == 'mean', 'Can only project mean attributes.'
        projections = project(nodes, vis_dims, maximise=maximise, resolution=project_resolution)
        colour_dim = attribute[1]
        if type(colour_dim) == str: colour_dim = tree.root.source.dim_names.index(colour_dim)
        # Ensure slice_dict is factored into the weighting.
        weight_dims = vis_dims.copy()
        if slice_dict != {}:
            for d, s in enumerate(slice_list): 
                if s is not None and type(s) not in (float, int) and d not in vis_dims:
                    weight_dims.append(d)
                    for i in range(len(projections)):
                        projections[i][0] = np.vstack((projections[i][0], s))
        values = [weighted_average(leaves, colour_dim, bb, weight_dims) for bb,leaves in projections]
        bbs = [bb_clip(bb[:len(vis_dims)], tree.root.bb_min[vis_dims]) for bb,_ in projections]
    else:
        # If don't require projection.
        values = list(gather_attributes(nodes, [attribute])[0])
        bbs = [(bb_clip(node.bb_max[vis_dims], tree.root.bb_min[vis_dims]) if maximise 
                else node.bb_min[vis_dims]) for node in nodes]
    # Create rectangles.
    lims_and_values_to_rectangles(ax, bbs, 
        values=values, cmap=_cmap(attribute), cmap_lims=cmap_lims, 
        fill_colour=fill_colour, edge_colour=edge_colour)
    return ax

def show_difference_rectangles(tree_a, tree_b, attribute, max_depth=np.inf, maximise=False, cmap_lims=None, edge_colour=None, ax=None):
    """
    Given two trees with the same two split_dims, display rectangles coloured by the differences in the given attribute.
    TODO: Adapt for slicing. 
    """
    raise NotImplementedError("Need to clip intersection to *inner* of tree_a,root.bb_min and tree_b,root.bb_min")
    assert len(tree_a.split_dims) == 2 and tree_a.split_dims == tree_b.split_dims
    vis_dims = tree_a.split_dims
    # Set up axes.
    ax = _ax_setup(ax, tree_a, vis_dims, attribute=attribute, diff=True, tree_b=tree_b)    
    # Collect the lists of nodes to show.
    nodes_a = list(tree_a.propagate({}, mode=('max' if maximise else 'min'), max_depth=max_depth))
    values_a = gather_attributes(nodes_a, [attribute])
    nodes_b = list(tree_b.propagate({}, mode=('max' if maximise else 'min'), max_depth=max_depth))
    values_b = gather_attributes(nodes_b, [attribute])
    # Compute the pairwise intersections between nodes.
    intersections = []; diffs = []
    for node_a, value_a in zip(nodes_a, values_a[0]):
        for node_b, value_b in zip(nodes_b, values_b[0]):
            inte = bb_intersect(node_a.bb_max[vis_dims] if maximise else node_a.bb_min[vis_dims], 
                                node_b.bb_max[vis_dims] if maximise else node_b.bb_min[vis_dims])
            if inte is not None: # Only store if intersection is non-empty.
                intersections.append(inte)
                diffs.append(value_a - value_b)
    # Create rectangles.
    lims_and_values_to_rectangles(ax, intersections, values=diffs, cmap=_cmap(attribute), cmap_lims=cmap_lims, edge_colour=edge_colour)    
    return ax

def lims_and_values_to_rectangles(ax, lims, offsets=None, values=None, cmap=None, cmap_lims=None, fill_colour=None, edge_colour=None):
    """xxx"""
    if values != [None]:
        # Compute fill colour.
        if cmap_lims is None: mn, mx = np.min(values), np.max(values)
        else: mn, mx = cmap_lims
        if mx == mn: fill_colours = [cmap[0](0.5) for _ in values] # Default to midpoint.
        else: fill_colours = [cmap[0](v) for v in (np.array(values) - mn) / (mx - mn)]
        # Create colour bar.
        norm = mpl.colors.Normalize(vmin=mn, vmax=mx)
        ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap[1]), ax=ax)
    else:
        if fill_colour == None and edge_colour == None: edge_colour = 'k' # Show lines by default if no fill.    
        fill_colours = [fill_colour for _ in lims]
    for i, (l, fill_colour) in enumerate(zip(lims, fill_colours)):
        r = _make_rectangle(l, fill_colour, edge_colour, alpha=1)
        ax.add_patch(r)
        if offsets is not None: # For 3D plotting.
            art3d.pathpatch_2d_to_3d(r, z=offsets[i], zdir="z")
    ax.autoscale_view()

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
    values = gather_attributes(list(tree.propagate({}, mode=('max' if maximise else 'min'), max_depth=max_depth)), attributes)
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
        pos["I"] = np.array([0, 1.6])
        pos["T"] = np.array([0, -0.1])
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
    else: edge_colours = ["k" for _ in G.edges]
    arcs = nx.draw_networkx_edges(G, pos=pos, connectionstyle="arc3,rad=0.2", edge_color=edge_colours)
    # Set alpha individually for each non-highlighted edge.
    if alpha:
        for arc, (_,_,attr), c in zip(arcs, G.edges(data=True), edge_colours):
            if c != "r": arc.set_alpha(attr["alpha"])
    # Retrieve axis ticks which networkx likes to hide.
    if layout_dims is not None: 
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    return ax

def show_shap_dependence(tree, node, wrt_dim, shap_dim, vis_dim=None, deinteraction_dim=None, 
                         colour_dim=None, colour='k', subsample=None):
    """
    For all the samples at a node (or a subsample), scatter the SHAP values for shap_dim w.r.t. wrt_dim.
    Distribute points along shap_dim *or* a specified vis_dim, and optionally colour points by colour_dim.
    TODO: Remove interaction effects with "deinteraction_dim"
    """
    # Allow dim_names to be specified instead of numbers.
    if vis_dim is None: vis_dim = shap_dim
    if type(shap_dim) == str: shap_dim = tree.root.source.dim_names.index(shap_dim)
    if type(wrt_dim) == str: wrt_dim = tree.root.source.dim_names.index(wrt_dim)
    if type(vis_dim) == str: vis_dim = tree.root.source.dim_names.index(vis_dim)
    if type(deinteraction_dim) == str: deinteraction_dim = tree.root.source.dim_names.index(deinteraction_dim)
    if type(colour_dim) == str: colour_dim = tree.root.source.dim_names.index(colour_dim)
    # Compute SHAP values for all samples.
    X = node.source.data[subsample_sorted_indices(node.sorted_indices, subsample)[:,0]]
    if deinteraction_dim is None: 
        shaps = tree.shap(X, wrt_dim=wrt_dim, maximise=False)
        d = np.array(list(zip(X[:,vis_dim], 
                          [s[shap_dim] for s in shaps])))
    else: 
        # Remove interaction effects with deinteraction_dim.
        shaps = tree.shap_with_ignores(X, wrt_dim=wrt_dim, ignore_dims=[deinteraction_dim], maximise=False)
        d = np.array(list(zip(X[:,vis_dim], 
                          [s[shap_dim] - (i[shap_dim] / 2) for s,i in # <<< NOTE: DIVIDE BY 2?
                          zip(shaps[None], shaps[deinteraction_dim])])))





    if colour_dim is not None: c = X[:,colour_dim]
    # Set up figure.
    _, ax = mpl.pyplot.subplots(figsize=(12/5,12/5))
    ax.set_xlabel(tree.root.source.dim_names[vis_dim])    
    ax.set_ylabel(f'SHAP for {tree.root.source.dim_names[shap_dim]} w.r.t. {tree.root.source.dim_names[wrt_dim]}')
    if colour_dim is None: colours = colour
    else:
        # cmap = (mpl.cm.copper,'copper')
        cmap = (mpl.cm.coolwarm_r, 'coolwarm_r') 
        mn, mx = np.min(c), np.max(c)
        if mx == mn: fill_colours = [cmap[0](.5) for _ in c]
        else: colours = [cmap[0](v) for v in (c - mn) / (mx - mn)]
        # Create colour bar.
        norm = mpl.colors.Normalize(vmin=mn, vmax=mx)
        cbar = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap[1]), ax=ax)
        cbar.set_label(tree.root.source.dim_names[colour_dim], rotation=270)
    ax.scatter(d[:,0], d[:,1], s=0.5, alpha=0.075, c=colours)
    return ax

def _ax_setup(ax, tree, vis_dims, attribute=None, diff=False, tree_b=None, derivs=False, slice_dict={}):
    if ax is None: _, ax = mpl.pyplot.subplots(figsize=(3,12/5))
    vis_dim_names = [tree.root.source.dim_names[v] for v in vis_dims]
    ax.set_xlabel(vis_dim_names[0]); ax.set_ylabel(vis_dim_names[1])
    title = tree.name
    if diff: title += f' vs {tree_b.name}\n$\Delta$ in {attribute[0]} of {attribute[1]}'
    elif attribute: title += f'\n{attribute[0]} of {attribute[1]}'
    elif derivs: title += '\nTime derivatives'
    if slice_dict != {}: title += '\nSlice at '+', '.join([f'{d} = {v}' for d, v in slice_dict.items()])
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

def _make_rectangle(lims, fill_colour, edge_colour, alpha):
    (xl, xu), (yl, yu) = lims
    fill_bool = (fill_colour != None)
    return mpl.patches.Rectangle(xy=[xl,yl], width=xu-xl, height=yu-yl, fill=fill_bool, facecolor=fill_colour, alpha=alpha, edgecolor=edge_colour, lw=0.5, zorder=-1) 

def _cmap(attribute):
    if attribute is None: return None
    if attribute[0] in ('std','std_c','iqr'): return (mpl.cm.coolwarm, 'coolwarm') # Reverse for measures of spread.
    else:                                     return (mpl.cm.coolwarm_r, 'coolwarm_r')                 