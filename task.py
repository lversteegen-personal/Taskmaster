class task:

    def __init__(self, n_actions, action, check_completed):

        self.n_actions = n_actions
        self.action = action
        self.check_completed = check_completed

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