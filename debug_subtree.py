from joblib import load
import hyperrectangles as hr

tree = load("tree.joblib")
x = {k:v for k,v in zip(tree.space.dim_names[3:11], 

[0.1,1,
0,-0.6,
0,0.4,
0,0]

)}
print(x)

# hr.diagram(tree, tree.eval_dims, sf=3, verbose=False, png=True)

leaves = tree.propagate(x, mode="maximise")

subtree = tree.dca_subtree("subtree", leaves)

hr.diagram(subtree, subtree.eval_dims, sf=3, verbose=False, png=True)