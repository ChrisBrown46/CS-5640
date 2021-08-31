import numpy as np


class RandomAgent(object):
    def __init__(self, environment):
        self.environment = environment
        self.value_table = "A random agent has no knowledge"

    def act(self, _):
        return self.environment.action_space.sample()

    def learn(self, state, next_state, action, reward):
        return


class TabularAgent(object):
    def __init__(self, environment):
        self.environment = environment
        self.explore_rate = 0.1

        """
        Initialize a table to hold an expected value for every state.
            The table will be 3-dimensional for this problem.
            The first dimension will hold the cars position,
            The second dimension will hold the cars velocity,
            The third dimension will hold values for each action in the current state.
        A state's value will be accessed by:
            state_value = table[position, velocity, action]
            state_value = table[(position, velocity), action]
            state_value = table[(state), action]
            state_value = table[state, action]
        The table below has 100 "buckets" for position and velocity, and 3 possible actions.
            Adjust this table's size as you see fit. A larger table allows greater precision, but less generalization.
            You will also have to decide to how partition states to their buckets.
                Linear scaling? Logistic scaling? Log scaling? Sigmoid scaling? etc.
        """
        self.value_table = np.zeros(shape=(100, 100, 3))

    def act(self, state):
        if np.random.random() < self.explore_rate:
            action = self.environment.action_space.sample()
        else:
            action = np.argmax(self.value_table[state])
        return action

    def learn(self, state, next_state, action, reward):
        """
        Simple learning function that says "if the next state is good, this state must also be good."
        """
        self.value_table[state, action] += np.average(self.value_table[next_state])
        self.value_table[state, action] += reward
