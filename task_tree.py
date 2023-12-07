import numpy as np
from task import task

class task_tree_node:

    def __init__(self, task : task, state, depth):

        self.task = task
        self.state = state
        self.completed = task.check_completed(state)
        self.expanded = False
        self.depth = depth

    def try_action(self, action_code):

        if not self.expanded:
            raise RuntimeError("This node has not been expanded.")

        if self.invalid_actions[action_code]:
            return False
        elif self.children[action_code] != None:
            return True

        success, result = self.task.action(self.state,action_code)
        if success:
            self.children[action_code] = task_tree_node(self.task,result,self.depth+1)
            self.copy_state = self.state.copy()
            return True
        else:
            self.invalid_actions[action_code] = True
            return False

    def expand(self, input, initial_evaluation, initial_policy):

        if self.completed:
            raise RuntimeError("This state is completed and should not have been evaluated.")
        elif self.expanded:
            raise RuntimeError("This state is already expanded and should not have been evaluated.")

        self.expanded = True
        self.input = input
        self.initial_evaluation = initial_evaluation
        self.initial_policy = initial_policy
        self.children = [None]*self.task.n_actions
        self.invalid_actions = np.zeros(self.task.n_actions,dtype=bool)
