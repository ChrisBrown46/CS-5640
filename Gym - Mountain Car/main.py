from agents import RandomAgent
import gym
import numpy as np

env = gym.make("MountainCar-v0")
state = env.reset()

agent = RandomAgent(env)

for iteration in range(1):

    state = env.reset()
    done = False
    while not done:
        env.render()

        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, next_state, action, reward)
        state = next_state

    env.render()
env.close()

"""
MountainCar Movements
    0 - Move left
    1 - Don't move
    2 - Move right
"""
print(agent.value_table)
