# Local imports
from os import environ
from agents import *

# OpenAI Gym imports
import gym

environment = gym.make("PongDeterministic-v4")
agent = DeepQualityNetworkAgent(environment)

# TODO: Implement resuming training
# takes
#   memory
#   network weights
#   exploration rate
#   iteration

best_score = -21.0
for iteration in range(10_000):
    done = False
    state = environment.reset()
    state = agent.convert_state(state)

    steps = 0

    while not done:
        # environment.render()

        action = agent.act(state)
        next_state, reward, done, _ = environment.step(action)

        next_state = agent.convert_state(next_state)
        agent.learn(state, next_state, action, reward, done)

        state = next_state
        steps += 1

    best_score = max(best_score, agent.total_reward)
    print(
        f"""Iteration: {iteration}
    Exploration Rate: {agent.exploration_rate:.7f}
    Steps: {steps}
    Score: {agent.total_reward}
    Best-Score: {best_score}
    """
    )
    agent.finish_iteration(iteration)

environment.close()
