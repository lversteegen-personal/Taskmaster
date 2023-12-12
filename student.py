import numpy as np
from task_tree import task_tree_node
from evaluation_tree import evaluation_tree_node

from task import task

class student:

    def __init__(self, task:task, n_sim, neural_network):

        self.task = task
        self.n_actions = task.n_actions
        self.n_sim = n_sim

        self.neural_network = neural_network

    def expand_task_nodes(self, task_nodes:list):

        #Make elements unique
        s = set(task_nodes)
        task_nodes = [t for t in s if not (t.completed or t.expanded)]

        eval,policy = self.neural_network.predict_value(task_nodes)

        for i, t in enumerate(task_nodes):

            t.expand(eval[i],policy[i])

    def run_simulation_step(self, eval_trees):

        eval_leaves = []
        t : evaluation_tree_node
        for t in eval_trees:

            eval_leaves.append(t.find_leaf())
        
        self.expand_task_nodes(eval_leaves)

        for i, t in enumerate(eval_trees):

            if not t.completed:
                t.update(eval_leaves[i])

    def run_action_step(self, task_nodes):

        t:task_tree_node
        to_expand = []
        for t in task_nodes:
            if t.completed:
                raise RuntimeError("Cannot start from completed proof node")
            elif not t.expanded:
                to_expand.append(t)

        self.expand_task_nodes(to_expand)

        initial_eval_trees = [evaluation_tree_node(t) for t in task_nodes]
        eval_trees = set(initial_eval_trees)

        for i in range(self.n_sim):

            self.run_simulation_step(eval_trees)
            eval_trees.difference_update([e for e in eval_trees if e.completed])

        result_list = []

        for i,t in enumerate(task_nodes):
            
            a, pi = initial_eval_trees[i].select_action()
            if not t.try_action(a):
                raise ValueError("This action is not valid.")
            else:
                result_list.append((a,pi, initial_eval_trees[i]))

        return result_list
    