from neural_networks import build_dqn, clone_model, resize_frame

import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class RandomAgent(object):
    def __init__(self, environment):
        self.environment = environment

    def act(self, *_):
        return self.environment.action_space.sample()

    def convert_state(self, state):
        return state

    def learn(self, *_):
        return


class DeepQualityNetworkAgent(RandomAgent):
    def __init__(self, environment):
        self.environment = environment
        self.possible_actions = [0, 2, 3]

        # Learning updates
        self.learning_rate = 0.001
        self.discount = 0.95

        # Exploration
        self.min_exploration_rate = 0.005
        self.exploration_rate = 1.05  # slightly bigger so we don't get greedy too soon
        self.exploration_decay = 0.999

        # Custom neural network; every X iterations update the target
        self.model = build_dqn(self.possible_actions, self.learning_rate)
        self.target_model = clone_model(self.model)
        self.model_update_frequency = 100

        # Batching update
        self.batch_size = 1024
        self.frames = deque(maxlen=4)

        # Memory replay
        self.memories = deque(maxlen=25_000)

        # Plotting variables
        self.reward_list = []
        self.average_reward_list = []
        self.total_reward = 0
        self.plotting_iterations = 100
        self.image_path = "./plotting_images"

        # Folders for storing plots
        if os.path.exists(self.image_path):
            shutil.rmtree(self.image_path)
        os.mkdir(self.image_path)

    def act(self, state):
        if np.random.random() < self.exploration_rate:
            return np.random.choice(self.possible_actions, 1)[0]
        else:
            actions = self.model.predict(state[np.newaxis, ...])
            action_index = np.argmax(actions)
            return self.possible_actions[action_index]

    def convert_state(self, state):
        # Convert the frame into an 84x84 image
        frame = resize_frame(state)
        self.frames.append(frame)

        # Fill the frame buffer with dummy data if we restart the game
        while len(self.frames) < 4:
            self.frames.append(frame)

        # Stack 4 frames into one state
        state = np.empty(shape=(4, 84, 84), dtype=float)
        for offset in range(4):
            state[offset] = self.frames[offset] / 255

        # Convert (4, 84, 84) -> (84, 84, 4)
        state = np.moveaxis(state, 0, 2)

        return state

    def learn(self, state, next_state, action, reward, done, *_):
        self.total_reward += reward
        self.memories.append([state, next_state, action, reward, done])

        if done:
            self.memory_replay()

    # Perform a TD update on every memory
    def memory_replay(self):
        n = min(self.batch_size, len(self.memories))
        states, next_states, actions, rewards = self.batch_memories(n)

        state_qualities = self.model.predict(states)
        next_state_qualities = self.target_model.predict(next_states)

        for index in range(n):
            self.td_update(
                actions, rewards, index, state_qualities, next_state_qualities
            )

        self.model.fit(states, state_qualities, batch_size=self.batch_size, verbose=0)

    def batch_memories(self, n):
        np.random.shuffle(self.memories)

        states = np.empty(shape=(n, 84, 84, 4), dtype=float)
        next_states = np.empty(shape=(n, 84, 84, 4), dtype=float)
        actions = np.empty(shape=(n, 1), dtype=int)
        rewards = np.empty(shape=(n, 1), dtype=int)

        for index in range(n):
            state, next_state, action, reward, done = self.memories[index]

            states[index] = state
            next_states[index] = next_state
            actions[index] = self.possible_actions.index(action)
            rewards[index] = reward + (not done)
        return states, next_states, actions, rewards

    def td_update(self, actions, rewards, index, state_qualities, next_state_qualities):
        action = actions[index]
        update = rewards[index] + self.discount * np.amax(next_state_qualities[index])
        state_qualities[index][action] = update

    # Exploration decay and plotting
    def finish_iteration(self, iteration):
        # Exploration rate adjustments
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)

        # Update the target model every so often
        if iteration % self.model_update_frequency == 0:
            self.target_model.set_weights(self.model.get_weights())
            self.model.save(
                f"models/model_snapshot_{iteration}_{self.exploration_rate:.3f}"
            )

        # Plotting below
        self.reward_list.append(self.total_reward)
        self.total_reward = 0

        if len(self.reward_list) > self.plotting_iterations:
            self.average_reward_list.append(
                np.mean(self.reward_list[: -self.plotting_iterations])
            )
        else:
            self.average_reward_list.append(np.mean(self.reward_list))

        if (iteration + 1) % self.plotting_iterations == 0:
            self.plot(iteration + 1)

    # Currently in construction; check back later :)
    def plot(self, iteration):
        plt.figure(figsize=(20, 4), facecolor="white")
        plt.title(f"Iteration {iteration}")
        plt.plot(np.arange(len(self.reward_list), dtype=int), self.reward_list, c="k")
        plt.plot(
            np.arange(len(self.reward_list), dtype=int),
            self.average_reward_list,
            c="r",
            linewidth=2,
        )
        plt.xlabel("iterations")
        plt.ylabel("reward")
        plt.ylim([-22, 22])

        file_name = f"{self.image_path}/{iteration}.png"
        plt.savefig(file_name)
        plt.close("all")
