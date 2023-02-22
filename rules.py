from .utils import *
from .visualise import _values_to_colours
import numpy as np
import pydot 
import io
import matplotlib.image as mpimg
from matplotlib.cm import coolwarm_r
from matplotlib.colors import rgb2hex


def rules(tree, pred_dims=None, sf=3, dims_as_indices=False, out_name=None):
    """
    Represent tree as a rule set with pred_dims as the consequent. Formatted as valid Python code.
    """
    pred_dims = tree.space.idxify(pred_dims)
    lines = []
    def _recurse(node, depth=0):
        i = "    " * (depth+1) # Indent.     
        if node is None: lines.append(f"{i}return None")       
        elif node.split_dim is not None:
            dim_name = tree.space.dim_names[node.split_dim]
            if dims_as_indices:
                dim_text, comment = f"x[{node.split_dim}]", f" # {dim_name}"
            else: 
                dim_text, comment = dim_name, ""
            lines.append(f"{i}if {dim_text} < {round_sf(node.split_threshold, sf)}:{comment}")
            _recurse(node.left, depth+1)
            lines.append(f"{i}else:")
            _recurse(node.right, depth+1)
        else: 
            if pred_dims:
                lines.append(f"{i}return {round_sf(node.mean[pred_dims], sf)} # n={node.num_samples}, std={round_sf(np.sqrt(np.diag(node.cov)[pred_dims]), sf)}")
            else: lines.append(f"{i}return # n={node.num_samples}")
    _recurse(tree.root)
    lines.insert(0, f"def {tree.name}(x):")
    if out_name is not None:  # If out_name specified, write out.
        with open(out_name, "w", encoding="utf-8") as f:
            for l in lines: f.write(l+"\n")
    return "\n".join(lines)

def diagram(tree, pred_dims=None, colour_dim=None, cmap_lims=None,
            show_decision_node_preds=False, show_num_samples=False, show_std_rng=False, show_impurity=False, add_to_leaf_nums=0,
            sf=3, out_as="svg", out_name=None, size=None):
    """
    Represent tree as a pydot diagram with pred_dims and the consequent.
    """
    if pred_dims is not None: pred_dims = pred_dims = tree.space.idxify(pred_dims)
    if colour_dim is not None:
        colour_dim = tree.space.idxify(colour_dim)
        leaf_means = tree.gather(("mean", colour_dim))
        if cmap_lims is None: cmap_lims = (min(leaf_means), max(leaf_means))
        colour = lambda node: rgb2hex(_values_to_colours([node.mean[colour_dim]], (coolwarm_r, "coolwarm_r"), cmap_lims, None, False)[0])
    graph_spec = 'digraph Tree {nodesep=0.2; ranksep=0.25; node [shape=box, penwidth=1.107];'
    def _recurse(node, graph_spec, n=0, n_parent=0, dir_label=None):
        if node is None: graph_spec += f'{n} [label="None"];'
        else:
            if node.split_dim is None:
                c = colour(node) if colour_dim is not None else "white"
                leaf_num = tree.leaves.index(node)+add_to_leaf_nums
            else:
                c = colour(node) if (colour_dim is not None and show_decision_node_preds) else "white"
                split = f'{tree.space.dim_names[node.split_dim]}≥{round_sf(node.split_threshold, sf)}'
            graph_spec += f'{n} [style=filled, fontname="ETBembo", fontsize=16.5, height=0, width=0, fillcolor="{c}", label="'
            if node.split_dim is None or show_decision_node_preds:
                if node.split_dim is None: graph_spec += f'{leaf_num} '
                if pred_dims:
                    for d, (mean, std, rng) in enumerate(zip(node.mean[pred_dims], np.sqrt(np.diag(node.cov)[pred_dims]), node.hr_min[pred_dims])):
                        graph_spec += f'{tree.space.dim_names[pred_dims[d]]}={round_sf(mean, sf)}'
                        if show_std_rng: graph_spec += f' (s={round_sf(std, sf)},r={round_sf(rng, sf)})'
                if show_num_samples: graph_spec += f'\nn={node.num_samples}'
                if pred_dims and show_impurity:
                    imp = f"{np.dot(node.var_sum[pred_dims], tree.space.global_var_scale[pred_dims]):.2E}"
                    graph_spec += f'\nimpurity: {imp}'
                if node.split_dim is not None:
                    graph_spec += f'\n-----\n{split}'
            else: 
                graph_spec += f'{split}'
            graph_spec += '"];'
            n_here = n
            if n_here > 0: # Make edge from parent.
                graph_spec += f'{n_parent} -> {n} [penwidth=1.107, arrowsize=1.107'
                if dir_label is not None: graph_spec += 'fontname="ETBembo", fontsize=16.5, label=" {dir_label} "' # Spaces add padding
                graph_spec += '];'
            n += 1
            if node.split_dim is not None: # Recurse to children.
                graph_spec, n = _recurse(node.left, graph_spec, n, n_here, dir_label=None)
                graph_spec, n = _recurse(node.right, graph_spec, n, n_here, dir_label=None)
        return graph_spec, n
    # Create and save pydot graph.    
    graph_spec, _ = _recurse(tree.root, graph_spec)
    (graph,) = pydot.graph_from_dot_data(graph_spec+'}') 
    if size is not None: graph.set_size(f"{size[0]},{size[1]}!")
    if out_as == "png":   graph.write_png(f"{out_name if out_name is not None else tree.name}.png") 
    elif out_as == "svg": graph.write_svg(f"{out_name if out_name is not None else tree.name}.svg") 
    elif out_as == "plt": # https://stackoverflow.com/a/18522941
        png_str = graph.create_png()
        sio = io.BytesIO()
        sio.write(png_str)
        sio.seek(0)
        return mpimg.imread(sio)
    else: raise ValueError()

def rule(node, maximise=True, sf=3): 
    """
    Describe the hyperrectangle for a node.
    """
    return _rule_inner(node.hr_max if maximise else node.hr_min, node.space.dim_names, sf)

def _rule_inner(hr, dim_names, sf):
    terms = []
    for i, (mn, mx) in enumerate(hr):
        do_mn, do_mx = mn != -np.inf, mx != np.inf
        if do_mn and do_mx:
            terms.append(f"{round_sf(mn, sf)} =< {dim_names[i]} < {round_sf(mx, sf)}")
        else:
            if do_mn: terms.append(f"{dim_names[i]} >= {round_sf(mn, sf)}")
            if do_mx: terms.append(f"{dim_names[i]} < {round_sf(mx, sf)}")
    return " and ".join(terms)

def difference_rule(node_a, node_b, maximise=True, sf=3):
    """
    Describe the difference between the hyperrectangles for two nodes.
    TODO: Refactor
    """
    dn = node_a.space.dim_names
    def _extend(hr):
        hr[hr[:,0]==mbb[:,0],0], hr[hr[:,1]==mbb[:,1],1] = -np.inf, np.inf; return hr
    hr_a = node_a.hr_max if maximise else node_a.hr_min
    hr_b = node_b.hr_max if maximise else node_b.hr_min
    mbb = hr_mbb(hr_a, hr_b) # MBB sets the context for the description
    return f"While {_rule_inner(mbb, dn, sf)}, ({_rule_inner(_extend(hr_a), dn, sf)}) --> ({_rule_inner(_extend(hr_b), dn, sf)})"

def counterfactual(x, options, delta_dims, sf=3): 
    """
    Describe a set of counterfactual options.
    """
    if type(options) == tuple: options = [options]
    delta_dims = options[0][0].space.idxify(delta_dims)
    dim_names, operators, options_text = options[0][0].space.dim_names, [None,">=","<"], []    
    for leaf, x_closest, l0, l2 in options:
        terms = []
        for d, diff in enumerate(np.sign(x_closest - x)):
            if d in delta_dims and diff:
                terms.append(f"{dim_names[d]} {operators[int(diff)]} {round_sf(x_closest[d], sf)}")            
        if terms:
            options_text.append(" and ".join(terms) + f" (l0 = {int(l0)}, weighted l2 = {round_sf(l2, sf)})")
        else: options_text.append("Foil already satisfied.")
    return "\nor\n".join(options_text)

def temporal(): 
    """
    Should call counterfactual().
    """
    raise NotImplementedError()
