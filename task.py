from typing import Callable

class task:

    def __init__(self, input_size:int, n_actions:int, action:Callable, check_completed:Callable, reward_function:Callable, make_input:Callable):

        self.input_size = input_size
        self.n_actions = n_actions
        self.action = action
        self.check_completed = check_completed
        self.reward_function = reward_function
        self.make_input = make_input

class setup:

    def __init__(self, initial_state, n_actions, action):

        self.initial_state = initial_state
        self.n_actions = n_actions
        self.action = action

    # def task_action(self, states, action_codes):

    #     return self.task_action(states,action_codes)

    # def setup_action(self, states, action_codes):

    #     return self.setup_action(states,action_codes)

    # def check_complete(self, states):

    #     return self.check_complete(states)