import gym
import numpy as np
import cv2

from TreeConvolution.util import prepare_trees

from .space import Space
from .rules import diagram

gym.envs.registration.register(
    id="TreeGrower-v0", 
    entry_point="hyperrectangles.env:TreeGrower",
	)

class TreeGrower(gym.Env):
    """
    Cast the growth of a regression tree as a reinforcement learning problem. 
    This is inspired by the following paper (although they consider classification):

    Wen, Guixuan, and Kaigui Wu. "Building Decision Tree for Imbalanced Classification via Deep Reinforcement Learning." 
    Asian Conference on Machine Learning. PMLR, 2021.
    """
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, data, dim_names, split_dims, eval_dims, render_mode=False): 
        self.space = Space(dim_names, data)
        self.split_dims, self.eval_dims = split_dims, eval_dims
        self.observation_space = None
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(len(self.split_dims)), gym.spaces.Box(np.float32([-1.]), np.float32([1.]))))
        # Set up rendering.
        self.render_mode = render_mode
        if self.render_mode: 
            assert self.render_mode in self.metadata["render.modes"]
            if self.render_mode == "human": cv2.namedWindow("TreeGrower-v0", cv2.WINDOW_NORMAL)
        self.ep = -1

    def reset(self): 
        self.ep += 1
        # TODO: Must start with a complete tree with randomly-initialised default attributes and thresholds.
        self.tree = self.space.tree_best_first(self.ep, self.split_dims, self.eval_dims, max_num_leaves=1, disable_pbar=True) 
        self.tree.split_queue = [self.tree.root]
        self.img = None
        return self.obs()

    def step(self, action): 
        # TODO: Add capability for accepting no-op action. 
        # This has to be done with care: observation must change after no-op is taken to avoid partial observability.
        assert action in self.action_space
        node = self.tree.split_queue.pop(0) # Breadth-first splitting.
        split_dim = action[0]
        # Interpret continuous action as a percentile to split at.
        split_index = int(round(len(node) * (action[1][0] + 1) / 2))
        if 0 < split_index < len(node): # TODO: This is hacky.
            node._do_split(split_dim, split_index=split_index)
            self.tree.leaves = self.tree._get_nodes(leaves_only=True)
            self.tree.split_queue += [node.left, node.right]
            # Reward is reduction in var_sum.
            e = self.tree.eval_dims
            reward = sum((node.var_sum[e] - (node.left.var_sum[e] + node.right.var_sum[e])) * self.space.global_var_scale[e])
        else: reward = 0.
        done = reward < 0 # Terminate if negative reward received.
        return self.obs(), reward, done, {}

    def obs(self):
        def transformer(node): 
            x = np.zeros(len(self.split_dims) + 1)
            raise NotImplementedError("How to handle leaf nodes? Percentile representation and use 0.5 for leaves?")
            if node.split_dim is not None:
                x[node.split_dim] = 1
                x[-1] = node.split_threshold
            return x
        def left_child(node): return node.left
        def right_child(node): return node.right
        return prepare_trees([self.tree.root], transformer, left_child, right_child)

    def render(self, mode="human", shape=(96*2, 96*4)): 
        assert mode == self.render_mode, f"Render mode is {self.render_mode}, so cannot use {mode}"
        img = np.ones((*shape, 3))
        dg = diagram(self.tree, out_as="img", size=(shape[1]/96, shape[0]/96))[:,:,:3] # 96ppi; ignore alpha channel.
        img[:dg.shape[0], :dg.shape[1]] = dg
        if self.render_mode == "human": cv2.imshow("TreeGrower-v0", img); cv2.waitKey(1)
        elif self.render_mode == "rgb_array": return img