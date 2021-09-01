# Local imports
from agents import RandomAgent

# Python imports
import numpy as np
from time import time  # just to have timestamps in the files

# OpenAI Gym imports
import gym
from gym import wrappers

# Remove the monitoring if you do not want a video
env = gym.make("MountainCar-v0")
env = wrappers.Monitor(env, "./videos/" + str(time()) + "/")

# Change the agent to a different one by simply swapping out the class
# ex) RandomAgent(env) --> TabularAgent(env)
agent = RandomAgent(env)

# We are only doing a single simulation. Increase 1 -> N to get more runs.
for iteration in range(1):

    # Always start the simulation by resetting it
    state = env.reset()
    done = False

    # Either limit the number of simulation steps via a "for" loop, or continue
    # the simulation until a failure state is reached with a "while" loop
    while not done:

        # Render the environment. You will want to remove this or limit it to the
        # last simulation iteration with a "if iteration == last_one: env.render()"
        env.render()

        # Have the agent
        #   1: determine how to act on the current state
        #   2: go to the next state with that action
        #   3: learn from feedback received if that was a good action
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, next_state, action, reward)

        # Progress to the next state
        state = next_state

    env.render()  # Render the last simluation frame.
env.close()  # Do not forget this! Tutorials like to leave it out.

"""
MountainCar Movements
    0 - Move left
    1 - Don't move
    2 - Move right
"""
print(agent.value_table)
