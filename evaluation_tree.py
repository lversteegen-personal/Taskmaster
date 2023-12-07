from task_tree import task_tree_node
import numpy as np
import math


class evaluation_tree_node:

    c = 1.0

    def __init__(self, task_node: task_tree_node):

        self.task_node: task_tree_node = task_node
        self.total_visits = 0
        self.n_actions = task_node.task.n_actions
        self.q = np.zeros(self.n_actions)
        self.n = np.zeros(self.n_actions)
        self.children = [None] * self.n_actions
        self.rng = np.random.default_rng(seed=0)
        self.completed = False

    def compute_current_policy(self):

        if not self.task_node.expanded:
            raise RuntimeError("Underlying state is not expanded")

        base_pi = self.task_node.initial_policy
        l = math.sqrt(1/(self.total_visits+1))

        a_min = np.max(self.q + l*base_pi)
        a_max = np.max(self.q + l)

        while abs(a_max-a_min) > 0.001:
            a = (a_min+a_max)/2
            c = np.sum(base_pi/(a-self.q)) * l
            if c > 1:
                a_min = a
            else:
                a_max = a

        a = (a_min+a_max)/2
        return base_pi/(a-self.q) * l

    def find_leaf(self):

        a = self.select_action()
        self.direct_to = a

        if self.children[a] != None:
            return self.children[a].find_leaf()
        else:
            return self.task_node.children[a]

    def select_action(self):

        if self.completed:
            return self.direct_to
        
        self.pi = self.compute_current_policy()
        self.pi[self.task_node.invalid_actions] = 0

        while True:

            self.pi /= self.pi.sum()
            a = self.rng.choice(self.n_actions, p=self.pi)

            if self.task_node.try_action(a):
                return a
            else:
                self.pi[a] = 0

    def update(self, proof_node: task_tree_node):

        if proof_node.completed:
            v = 1
            self.completed = True
        else:
            v = proof_node.initial_evaluation

        if self.children[self.direct_to] == None:
            if not proof_node.completed:
                self.children[self.direct_to] = evaluation_tree_node(proof_node)
            self.n[self.direct_to] = 1
            self.q[self.direct_to] = v
        else:
            self.children[self.direct_to].update(proof_node)
            self.q[self.direct_to] = (
                self.q[self.direct_to] * self.n[self.direct_to] + v)/(self.n[self.direct_to]+1)
            self.n[self.direct_to] += 1

        self.total_visits += 1
