from task import setup
import numpy as np

class teacher:

    def __init__(self, setup:setup, lamda):

        self.setup = setup
        self.rng = np.random.default_rng(seed=0)
        self.lamda = lamda

    def generate_problems(self, n):

        n_actions = self.setup.n_actions
        steps = 1+self.rng.poisson(self.lamda,size=n)
        states = []

        for i in range(n):
            s = self.setup.initial_state.copy()
            for j in range(steps[i]):
                _,s = self.setup.action(s,self.rng.choice(self.setup.n_actions))

            states.append(s)

        return states