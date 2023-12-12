from task import setup
import numpy as np

class teacher:

    def __init__(self, setup:setup, step_dist):

        self.setup = setup
        self.rng = np.random.default_rng(seed=0)
        self.step_dist = step_dist

    def generate_problems(self, n):

        n_actions = self.setup.n_actions
        steps = self.step_dist(n)
        states = []

        for i in range(n):
            s = self.setup.initial_state.copy()
            for j in range(steps[i]):
                _,s = self.setup.action(s,self.rng.choice(self.setup.n_actions))

            states.append(s)

        return states