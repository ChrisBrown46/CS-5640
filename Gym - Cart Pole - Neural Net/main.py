# Local imports
from agents import *

# OpenAI Gym imports
import gym
from gym import wrappers

environment = gym.make("CartPole-v1")
environment = wrappers.Monitor(
    environment,  # environment to watch
    f"./videos/",  # where to save videos
    force=True,  # clear old videos
    video_callable=lambda episode_id: episode_id == 2999,  # what run to record
)
agent = DeepQualityNetworkAgent(environment)

for iteration in range(3000):
    done = False
    state = environment.reset()
    steps = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = environment.step(action)
        agent.learn(state, next_state, action, reward, done)
        state = next_state
        steps += 1

    agent.finish_iteration(iteration)
    print(
        f"Iteration: {iteration}, Exploration Rate: {agent.exploration_rate:.7f}, Steps: {steps}"
    )

agent.make_animations()
environment.close()
