from .utils import *
import numpy as np

def rules(tree, pred_dim, sf=3, out_file=None): 
    """
    Represent tree as a rule set wuth pred_dim as the consequent.
    """
    if type(pred_dim) == str: pred_dim = tree.root.source.dim_names.index(pred_dim)
    d = tree.root.source.dim_names; lines = []
    def _recurse(node, depth=0):
        i = "    " * depth # Indent.     
        if node is None: lines.append(f"{i}return None")       
        elif node.split_dim is not None:
            lines.append(f"{i}if {d[node.split_dim]} < {round_sf(node.split_value, sf)}:")
            _recurse(node.left, depth+1)
            lines.append(f"{i}else:")
            _recurse(node.right, depth+1)
        else: lines.append(f"{i}return {round_sf(node.mean[pred_dim], sf)} (n={node.num_samples}, std={round_sf(np.sqrt(np.diag(node.cov)[pred_dim]), sf)})")
    _recurse(tree.root)
    if out_file is not None:  # If out_file specified, write
        with open(out_file+".py", "w", encoding="utf-8") as f:
            for l in lines: f.write(l+"\n")
    return "\n".join(lines)

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