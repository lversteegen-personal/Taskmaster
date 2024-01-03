from task_tree import task_tree_node
import numpy as np
import math
from scipy.stats import norm as normal_dist

class evaluation_tree_node:

    def __init__(self, task_node: task_tree_node, reward_discount):

        self.task_node: task_tree_node = task_node
        self.total_visits = 0
        self.n_actions = task_node.task.n_actions
        self.q = np.zeros(self.n_actions)
        self.n = np.zeros(self.n_actions)
        self.children = [None] * self.n_actions
        self.rng = np.random.default_rng(seed=0)
        self.completed = False

        self.pow = 1
        self.exploration_constant = 1

        self.value_network_trust = 0.5
        self.reward_discount = reward_discount

    def compute_reward_statistics(self):

        pi = self.task_node.initial_reward
        pi_v = self.task_node.reward_confidence ** 2 + 0.001

        #We take the variance of pi as an approximation for the variance for the value of each rollout
        q_v = pi_v / (0.001+self.n) / self.value_network_trust
        t = pi_v/(q_v+pi_v)
        t[self.n==0] = 0
        p_E = t*self.q+(1-t)*pi
        p_V = t**2*q_v+(1-t)**2*pi_v

        return p_E, p_V

    def find_leaf(self):

        if self.task_node.completed:
            return self.task_node

        a = self.select_exploration()
        self.direct_to = a

        if self.children[a] != None:
            return self.children[a].find_leaf()
        else:
            return self.task_node.children[a]

    def select_action(self):

        p_E,_ = self.compute_reward_statistics()
        p_E[self.task_node.invalid_actions] = 0

        while True:

            a = np.argmax(p_E)

            if self.task_node.try_action(a):
                return a, p_E
            else:
                p_E[a] = 0

    def select_exploration(self):

        p_E,p_V = self.compute_reward_statistics()
        p_E[self.task_node.invalid_actions] = 0

        while True:

            x = self.rng.normal(p_E, np.sqrt(p_V))
            x[self.task_node.invalid_actions] = -100
            a = np.argmax(x)

            if self.task_node.try_action(a):
                return a
            else:
                p_E[a] = 0

    def update(self, task_node: task_tree_node):

        if self.task_node.completed:
            if not task_node.completed:
                raise("Something went wrong here.")
            else:
                return

        if task_node.completed:
            v = task_node.task.reward_function(task_node.state)
        else:
            v = task_node.initial_evaluation

        v*=self.reward_discount**(task_node.depth-self.task_node.depth)

        if self.n[self.direct_to] == 0:
            self.children[self.direct_to] = evaluation_tree_node(task_node, self.reward_discount)
        else:
            self.children[self.direct_to].update(task_node)

        self.q[self.direct_to] = (self.q[self.direct_to] * self.n[self.direct_to] + v)/(self.n[self.direct_to]+1)
        self.n[self.direct_to] += 1
        self.total_visits += 1
