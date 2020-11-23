from .utils import *
import numpy as np
import pydot 

def rules(tree, pred_dims, sf=3, out_name=None): 
    """
    Represent tree as a rule set with pred_dims as the consequent.
    """
    if type(pred_dims[0]) == str: pred_dims = [tree.root.source.dim_names.index(p) for p in pred_dims]
    d = tree.root.source.dim_names; lines = []
    def _recurse(node, depth=0):
        i = "    " * depth # Indent.     
        if node is None: lines.append(f"{i}return None")       
        elif node.split_dim is not None:
            lines.append(f"{i}if {d[node.split_dim]} < {round_sf(node.split_threshold, sf)}:")
            _recurse(node.left, depth+1)
            lines.append(f"{i}else:")
            _recurse(node.right, depth+1)
        else: lines.append(f"{i}return {round_sf(node.mean[pred_dims], sf)} (n={node.num_samples}, std={round_sf(np.sqrt(np.diag(node.cov)[pred_dims]), sf)})")
    _recurse(tree.root)
    if out_name is not None:  # If out_name specified, write
        with open(out_name+".py", "w", encoding="utf-8") as f:
            for l in lines: f.write(l+"\n")
    return "\n".join(lines)

def diagram(tree, pred_dims, sf=3, verbose=False, colour="#ffffff", out_name=None, png=False):
    """
    Represent tree as a pydot diagram with pred_dims and the consequent.
    """
    if type(pred_dims[0]) == str: pred_dims = [tree.root.source.dim_names.index(p) for p in pred_dims]
    d = tree.root.source.dim_names; graph_spec = 'digraph Tree {node [shape=box];'
    def _recurse(node, graph_spec, n=0, n_parent=0, dir_label="<"):
        if node is None: graph_spec += f'{n} [label="None"];'
        else:   
            ns = node.num_samples  
            mean = round_sf(node.mean[pred_dims], sf)
            std = round_sf(np.sqrt(np.diag(node.cov)[pred_dims]), sf) 
            imp = f"{np.dot(node.var_sum[pred_dims], node.source.global_var_scale[pred_dims]):.2E}"
            if node.split_dim is not None:
                # Decision node.
                split = f'{d[node.split_dim]}={round_sf(node.split_threshold, sf)}'
                if verbose: graph_spec += f'{n} [label="mean: {mean}\nstd: {std}\nnum_samples: {ns}\nimpurity: {imp}\n-----\nsplit: {split}", style=filled, fillcolor="{colour}", fontname = "ETBembo"];'
                else: graph_spec += f'{n} [label="{split}", style=filled, fillcolor="{colour}", fontname = "ETBembo"];'
            else: 
                # Leaf node.
                graph_spec += f'{n} [label="mean: {mean}\nstd: {std}\nnum_samples: {ns}\nimpurity: {imp}", fontname = "ETBembo"];'
            n_here = n
            if n_here > 0: 
                # Edge.
                graph_spec += f'{n_parent} -> {n} [label="{dir_label}"];'
            n += 1
            if node.split_dim is not None: # Recurse to children.
                graph_spec, n = _recurse(node.left, graph_spec, n, n_here, "<")
                graph_spec, n = _recurse(node.right, graph_spec, n, n_here, "≥")
        return graph_spec, n
    graph_spec, _ = _recurse(tree.root, graph_spec)
    # Create and save pydot graph.    
    (graph,) = pydot.graph_from_dot_data(graph_spec+'}') 
    if png: graph.write_png(f"{out_name if out_name is not None else tree.name}.png") 
    else:   graph.write_svg(f"{out_name if out_name is not None else tree.name}.svg") 
    

def rule(node, maximise=True, sf=3): 
    """
    Describe the bounding box for one node.
    """
    d = node.source.dim_names; terms = []
    for i, (mn, mx) in enumerate(node.bb_max if maximise else node.bb_min):
        do_mn, do_mx = mn != -np.inf, mx != np.inf
        if do_mn and do_mx:
            terms.append(f"{round_sf(mn, sf)} < {d[i]} < {round_sf(mx, sf)}")
        else:
            if do_mn: terms.append(f"{d[i]} > {round_sf(mn, sf)}")
            if do_mx: terms.append(f"{d[i]} < {round_sf(mx, sf)}")
    return " and ".join(terms)

def counterfactual(): NotImplementedError()

def temporal(): 
    """
    Should call counterfactual().
    """
    raise NotImplementedError()