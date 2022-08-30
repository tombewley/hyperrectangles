from .utils import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

def show_samples(node, vis_dims, colour_dim=None, alpha=1, spark=False, subsample=None, cmap_lims=None, ax=None, cbar=True):
    """
    Scatter plot across vis_dims of all the samples contained in node.
    """
    assert len(vis_dims) in {2,3}
    vis_dims, colour_dim = node.space.idxify(vis_dims, colour_dim)
    X_all_dims = node.space.data[subsample_sorted_indices(node.sorted_indices, subsample)[:,0]]
    X = X_all_dims[:, vis_dims]
    lims = [[X[:,0].min(), X[:,0].max()], [X[:,1].min(), X[:,1].max()]]
    if ax is None: 
        if spark: ax = _ax_spark(ax, lims)
        elif len(vis_dims) == 3: fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")
        else: _, ax = plt.subplots()#figsize=(8,8))
    if colour_dim: 
        colours = _values_to_colours(X_all_dims[:,colour_dim].squeeze(), (mpl.cm.Reds_r, "Reds_r"), cmap_lims, ax, cbar)
    else: colours = "k"
    # Automatically calculate alpha.
    if alpha is None: alpha = 1 / len(X)**0.5
    if spark:
        ax.scatter(X[:,0], X[:,1], s=0.25, c=colours, alpha=alpha) 
        y = node.mean[vis_dims[1]]
        ax.plot(lims[0], [y,y], c="k")
    else:
        ax.set_xlabel(node.space.dim_names[vis_dims[0]])
        ax.set_ylabel(node.space.dim_names[vis_dims[1]])
        if len(vis_dims) == 3: 
            ax.scatter(X[:,0], X[:,1], X[:,2], s=5, c=colours, alpha=alpha)
            ax.set_zlabel(node.space.dim_names[vis_dims[2]])
        else:
            ax.scatter(X[:,0], X[:,1], c=colours, s=10, alpha=alpha, edgecolors="k", linewidth=1)
    return ax

def show_episodes(space, vis_dims, ep_indices=None, ax=None):
    """
    Show all samples in a space as per-episode line plots.
    """
    ep_dim, vis_dims = space.idxify("ep"), space.idxify(vis_dims)
    if ax is None: _, ax = plt.subplots(); ax.set_xlabel(space.dim_names[vis_dims[0]]); ax.set_ylabel(space.dim_names[vis_dims[1]])
    for ep in group_along_dim(space, ep_dim):
        n = ep[0,ep_dim]
        if ep_indices==None or n in ep_indices: 
            x, y = ep[:,vis_dims[0]], ep[:,vis_dims[1]]
            ax.plot(x, y, c="k", lw=2, zorder=2)
            ax.scatter(x[0], y[0], c="r", s=10, zorder=3) # Start and end markers.
            ax.scatter(x[-1], y[-1], c="g", s=10, zorder=3)
    return ax

def show_lines(model, attributes, vis_dim=None, max_depth=np.inf, maximise=False, show_spread=False, ax=None):
    """
    Given a one vis_dim, display one or more attributes using horizontal lines for each node.
    TODO: Adapt for slicing.
    """
    if vis_dim is None: 
        assert len(model.split_dims) == 1; vis_dim = model.split_dims[0] # Will fail if not a tree.
    else: vis_dim = model.space.idxify(vis_dim)
    num_attributes = len(attributes)
    if show_spread: 
        attributes += [(('std',a[1]) if a[0]=='mean' else 
                       (('q1q3',a[1]) if a[0]=='median' else None)) 
                       for a in attributes]
    # Collect the list of nodes to show.
    nodes = list(model.propagate({}, mode=('max' if maximise else 'min'), max_depth=np.inf))
    values = gather(nodes, *attributes)
    if len(attributes) == 1: values = [values]
    # Create new axes if needed.
    if ax is None: _, ax = plt.subplots()#figsize=(9,8))
    vis_dim_name = model.space.dim_names[vis_dim]
    ax.set_xlabel(vis_dim_name)
    # Colour cycle.
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, attr in enumerate(attributes[:num_attributes]):
        for n, (node, value) in enumerate(zip(nodes, values[i])):
            mn, mx = node.hr_max[vis_dim] if maximise else node.hr_min[vis_dim]
            mn = max(mn, model.root.hr_min[vis_dim,0])
            mx = min(mx, model.root.hr_min[vis_dim,1])
            ax.plot([mn,mx], [value, value], c=colours[i], label=(f'{attr[0]} of {attr[1]}' if n == 0 else None))
            if show_spread: # Visualise spread using a rectangle behind.
                if attr[0] == 'mean': # Use standard deviation.
                    std = values[i+num_attributes][n]
                    spread_mn, spread_mx = value - std, value + std      
                elif attr[0] == 'median': # Use quartiles.
                    spread_mn, spread_mx = values[i+num_attributes][n]
                ax.add_patch(_make_rectangle([[mn,mx],[spread_mn,spread_mx]], colours[i], edge_colour=None, alpha=0.25))
    ax.legend()
    return ax

def show_leaf_numbers(model, vis_dims, ax=None, fontsize=6):
    assert len(vis_dims) in {1,2}
    vis_dims = model.space.idxify(vis_dims)
    if ax is None: _, ax = plt.subplots()
    for n, l in enumerate(model.leaves): 
        ax.text(l.mean[vis_dims[0]], l.mean[vis_dims[1]] if len(vis_dims) > 1 else 0.5, n, ha="center", va="center", fontsize=fontsize)
    return ax

def show_rectangles(model, vis_dims=None, attribute=None, 
                    slice_dict=None, max_depth=np.inf, maximise=False, project_resolution=None,
                    vis_lims=None, cmap_lims=None, fill_colour=None, edge_colour=None, ax=None, cbar=True):
    """
    Compute the rectangular projections of nodes from model onto vis_dims, and colour according to attribute.
    Where multiple projections overlap, compute a marginal value using the weighted_average function from utils.
    """       
    if vis_dims is None: vis_dims = model.split_dims
    else: vis_dims = model.space.idxify(vis_dims)
    assert len(vis_dims) in {1,2}, "Must have |vis_dims| in {1,2}." # Will fail if not a tree.
    
    if vis_lims is not None: vis_lims = np.array(vis_lims) # Manually specify vis_lims.
    else: vis_lims = model.root.hr_min[vis_dims] # Otherwise use hr_min of root (for tree).
    # Set up axes.
    ax = _ax_setup(ax, model, vis_dims, attribute=attribute, slice_dict=slice_dict)
    # Collect the list of nodes to show.
    if slice_dict is None: slice_dict = {}
    if slice_dict != {}: slice_list = model.space.listify(slice_dict)
    else: slice_list = slice_dict
    nodes = list(model.propagate(slice_list, mode=('max' if maximise else 'min'), max_depth=max_depth))        
    try:
        # Check if can avoid doing projection.
        # TODO: This doesn't catch cases when nodes are non-overlapping despite vis_dims != split_dims.
        # if not np.array_equal(vis_dims, model.split_dims): assert not(attribute)  # Will fail if not a tree.            
        values = gather(nodes, attribute)
        hrs = [hr_intersect(node.hr_max[vis_dims] if maximise else node.hr_min[vis_dims],  vis_lims) for node in nodes]
    except:
        # Otherwise, projection required.
        assert attribute[0] == "mean", "Can only project mean attributes."
        assert maximise == False, "Can only project with hr_min"
        projections = project(nodes, vis_dims, maximise=maximise, resolution=project_resolution)
        colour_dim = model.space.idxify(attribute[1])
        # Ensure slice_dict is factored into the weighting.
        weight_dims = vis_dims.copy()
        if slice_dict != {}:
            for d, s in enumerate(slice_list): 
                if s is not None and type(s) not in (float, int) and d not in vis_dims:
                    weight_dims.append(d)
                    for i in range(len(projections)):
                        projections[i][0] = np.vstack((projections[i][0], s))        
        values = [weighted_average(leaves, colour_dim, hr, weight_dims) for hr, leaves in projections]
        hrs = [hr_intersect(hr[:len(vis_dims)], vis_lims) for hr,_ in projections]
    # Create rectangles.
    lims_and_values_to_rectangles(ax, hrs,
        values=values, cmap=_cmap(attribute), cmap_lims=cmap_lims, 
        fill_colour=fill_colour, edge_colour=edge_colour, cbar=cbar)
    return ax

def show_difference_rectangles(tree_a, tree_b, vis_dims, attribute, max_depth=np.inf, maximise=False, cmap_lims=None, edge_colour=None, ax=None, cbar=True):
    """
    Given two trees with the same two split_dims, display rectangles coloured by the differences in the given attribute.
    TODO: Adapt for slicing. 
    """
    if maximise: raise NotImplementedError("Need to clip hr_max intersection to *inner* of tree_a.root.hr_min and tree_b.root.hr_min")
    assert tree_a.space == tree_b.space
    vis_dims = tree_a.space.idxify(vis_dims)
    # Set up axes.
    ax = _ax_setup(ax, tree_a, vis_dims, attribute=attribute, diff=True, tree_b=tree_b)    
    # Collect the lists of nodes to show.
    nodes_a = list(tree_a.propagate({}, mode=('max' if maximise else 'min'), max_depth=max_depth))
    values_a = gather(nodes_a, attribute)
    nodes_b = list(tree_b.propagate({}, mode=('max' if maximise else 'min'), max_depth=max_depth))
    values_b = gather(nodes_b, attribute)
    # Compute the pairwise intersections between nodes.
    intersections = []; diffs = []
    for node_a, value_a in zip(nodes_a, values_a[0]):
        for node_b, value_b in zip(nodes_b, values_b[0]):
            inte = hr_intersect(node_a.hr_max[vis_dims] if maximise else node_a.hr_min[vis_dims],
                                node_b.hr_max[vis_dims] if maximise else node_b.hr_min[vis_dims])
            if inte is not None: # Only store if intersection is non-empty.
                intersections.append(inte)
                diffs.append(value_a - value_b)
    # Create rectangles.
    lims_and_values_to_rectangles(ax, intersections, values=diffs, cmap=_cmap(attribute), cmap_lims=cmap_lims, edge_colour=edge_colour, cbar=cbar)    
    return ax

def lims_and_values_to_rectangles(ax, lims, offsets=None, values=[None], cmap=None, cmap_lims=None, fill_colour=None, edge_colour=None, cbar=True):
    """
    Assemble a rectangle visualisation.
    """
    if values != [None]: fill_colours = _values_to_colours(values, cmap, cmap_lims, ax, cbar)
    else:
        if fill_colour == None and edge_colour == None: edge_colour = 'k' # Show lines by default if no fill.    
        fill_colours = [fill_colour for _ in lims]
    for i, (l, fill_colour) in enumerate(zip(lims, fill_colours)):
        r = _make_rectangle(l, fill_colour, edge_colour, alpha=1)
        ax.add_patch(r)
        if offsets is not None: # For 3D plotting.
            art3d.pathpatch_2d_to_3d(r, z=offsets[i], zdir="z")
    ax.autoscale_view()

def show_split_quality(node, figsize=(12,8), sharey=True):
    """
    Line plots of split quality values for node, across all split_dims.
    NOTE: Must have previously been stored by setting store_all_qual=True during splitting.
    """
    num_split_dims = len(node.all_qual)
    num_rows = int(np.floor(num_split_dims**.5)); num_cols = int(np.ceil(num_split_dims / num_rows))
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharey=sharey)
    axes = axes.flatten()
    for i, (d, q) in enumerate(node.all_qual.items()):
        non_nan = ~np.isnan(q)
        axes[i].set_title(node.space.dim_names[d])
        axes[i].plot(node.all_split_thresholds[d][non_nan], q[non_nan], c="k")
    for ax in axes[i+1:]: ax.axis("off")
    return axes

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
    values = gather(list(tree.propagate({}, mode=('max' if maximise else 'min'), max_depth=max_depth)), *attributes)
    # Create arrows centred at means.    
    plt.quiver(values[0], values[1], values[2], values[3], 
                      pivot=pivot, angles='xy', scale_units='xy', units='inches', 
                      color='k', scale=1/scale, width=0.02, minshaft=1)
    return ax

def show_shap_dependence(tree, node, wrt_dim, shap_dim, vis_dim=None, deint_dim=None, 
                         colour_dim=None, colour='k', alpha=1, subsample=None):
    """
    For all the samples at a node (or a subsample), scatter the SHAP values for shap_dim w.r.t. wrt_dim.
    Distribute points along shap_dim *or* a specified vis_dim, and optionally colour points by colour_dim.
    TODO: Remove interaction effects with "deint_dim"
    """
    if vis_dim is None: vis_dim = shap_dim
    shap_dim, wrt_dim, vis_dim, deint_dim, colour_dim = tree.space.idxify(shap_dim, wrt_dim, vis_dim, deint_dim, colour_dim)
    # Compute SHAP values for all samples.
    X = node.space.data[subsample_sorted_indices(node.sorted_indices, subsample)[:,0]]
    if deint_dim is None: 
        shaps = tree.shap(X, shap_dims=tree.split_dims, wrt_dim=wrt_dim, maximise=False)
        d = np.array(list(zip(X[:,vis_dim], 
                          [s[shap_dim] for s in shaps])))
    else: 
        # Remove interaction effects with deint_dim.
        shaps = tree.shap_with_ignores(X, shap_dims=tree.split_dims, wrt_dim=wrt_dim, ignore_dims=[deint_dim], maximise=False)
        d = np.array(list(zip(X[:,vis_dim], 
                          [s[shap_dim] - (i[shap_dim] / 2) for s,i in # <<< NOTE: DIVIDE BY 2?
                          zip(shaps[None], shaps[deint_dim])])))
    if colour_dim is not None: c = X[:,colour_dim]
    # Set up figure.
    _, ax = plt.subplots(figsize=(12/5,12/5))
    ax.set_xlabel(tree.space.dim_names[vis_dim])    
    ax.set_ylabel(f'SHAP for {tree.space.dim_names[shap_dim]} w.r.t. {tree.space.dim_names[wrt_dim]}')
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
        cbar.set_label(tree.space.dim_names[colour_dim], rotation=270)
    ax.scatter(d[:,0], d[:,1], s=0.05, alpha=alpha, c=colours)
    return ax

def _ax_setup(ax, model, vis_dims, attribute=None, diff=False, tree_b=None, derivs=False, slice_dict=None):
    if ax is None: _, ax = plt.subplots(figsize=(12,8))#(3,12/5))
    ax.set_xlabel(model.space.dim_names[vis_dims[0]])
    if len(vis_dims) == 1: ax.set_yticks([])
    else: ax.set_ylabel(model.space.dim_names[vis_dims[1]])
    title = model.name
    if diff: title += f' vs {tree_b.name}\n$\Delta$ in {attribute[0]} of {attribute[1]}'
    elif attribute: title += f'\n{attribute[0]} of {attribute[1]}'
    elif derivs: title += '\nTime derivatives'
    if slice_dict is not None: title += '\nSlice at '+', '.join([f'{d} = {v}' for d, v in slice_dict.items()])
    ax.set_title(title)
    return ax

def _ax_spark(ax, lims):
    if ax is None: _, ax = plt.subplots(figsize=(1.2,1.3))
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
    if len(lims) == 1: lims = [lims[0], [0, 1]] # Handle when only 1D lims specified.
    (xl, xu), (yl, yu) = lims
    fill_bool = (fill_colour != None)
    return mpl.patches.Rectangle(xy=[xl,yl], width=xu-xl, height=yu-yl, fill=fill_bool, facecolor=fill_colour, alpha=alpha, edgecolor=edge_colour, lw=0.5, zorder=-1) 

def _values_to_colours(values, cmap, cmap_lims, ax, cbar):
    # Compute fill colour.
    if cmap_lims is None: mn, mx = np.min(values), np.max(values)
    else: mn, mx = cmap_lims
    if mx == mn: colours = [cmap[0](0.5) for _ in values] # Default to midpoint.
    else: colours = [cmap[0](v) for v in (np.array(values) - mn) / (mx - mn)]
    if cbar:
        # Create colour bar.
        norm = mpl.colors.Normalize(vmin=mn, vmax=mx)
        ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap[1]), ax=ax)
    return colours

def _cmap(attribute):
    if attribute is None: return None
    if attribute[0] in ('std','std_c','iqr'): return (mpl.cm.coolwarm, 'coolwarm') # Reverse for measures of spread.
    else:                                     return (mpl.cm.Reds_r, 'Reds_r')
