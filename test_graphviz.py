import hypertree as ht
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(f"data/heuristic/continuous_train.csv", index_col=0)
source = ht.Source(df.values, list(df.columns))
# subset = source.subset(subsample=10000)
tree = source.best_first_tree('heuristic',
                              split_dims=['pos_x','pos_y','vel_x','vel_y','ang','vel_ang','left_contact','right_contact'], 
                              eval_dims=['main_engine','lr_engine'], 
                            #   sorted_indices=subset, 
                              max_num_leaves=5)

ht.diagram(tree, pred_dims=['main_engine','lr_engine'], colour="#cfceda")