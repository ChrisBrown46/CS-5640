from neural_networks import NeuralNetwork

import os, shutil, imageio
import numpy as np
import matplotlib.pyplot as plt


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
        self.action_space = environment.action_space.n
        self.observation_space = environment.observation_space.shape[0]

        # Learning hyperparameters
        self.learning_rate = 0.001
        self.discount = 0.95
        self.min_exploration_rate = 0.01
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995

        # Custom neural network: uses relu activations and mse loss
        self.dqn = NeuralNetwork()
        self.dqn.add_layer(units=self.observation_space)
        self.dqn.add_layer(units=24)
        self.dqn.add_layer(units=24)
        self.dqn.add_layer(units=self.action_space)

        # Memory replay variables
        self.max_memories = 5000
        self.memories = []

        # Plotting variables
        self.file_names = []
        self.trajectory = []
        self.reward_list = []
        self.average_reward_list = []
        self.total_reward = 0
        self.plotting_iterations = 25
        self.image_path = "./temp_images"
        if os.path.exists(self.image_path):
            shutil.rmtree(self.image_path)
        os.mkdir(self.image_path)

    def act(self, state):
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_space)
        else:
            actions = self.dqn.forward_propogation(state)
            return np.argmax(actions)

    def learn(self, state, next_state, action, reward, done, *_):
        self.total_reward += reward  # Plotting purposes
        self.td_update(state, next_state, action, reward)

        # The update is set to a temporary value (1.0) for now.
        # The real value is computed inside the memory replay.
        self.memories.append([1.0, state, next_state, action, reward])

        if done:
            self.memory_replay()

        if len(self.memories) > self.max_memories:
            self.clean_memories()

    def td_update(self, state, next_state, action, reward):
        q_values = self.dqn.forward_propogation(state)
        update = self.learning_rate * (
            reward
            + self.discount * np.amax(self.dqn.forward_propogation(next_state))
            - self.dqn.forward_propogation(state)[action]
        )
        q_values[action] += update
        self.dqn.fit(state, q_values)

        # For memory replay
        return update

    # Perform a TD update on every memory
    def memory_replay(self):
        np.random.shuffle(self.memories)
        new_memories = []

        for _, state, next_state, action, reward in self.memories:
            update = self.td_update(state, next_state, action, reward)
            new_memories.append([update, state, next_state, action, reward])

        self.memories = new_memories

    # Remove half of the memories so we can continue making more
    def clean_memories(self):
        # 1. Remove the oldest memories
        recent_memories = self.memories[int(len(self.memories) * 0.8) :]
        self.memories = self.memories[: int(len(self.memories) * 0.8)]

        # 2. Remove the least important memories
        important_memories = sorted(self.memories, key=lambda x: abs(x[0]))
        important_memories = important_memories[int(len(important_memories) * 0.8) :]
        self.memories = important_memories[: int(len(important_memories) * 0.8)]

        # 3. Randomly forget some memories
        self.memories = np.array(self.memories, dtype=object)
        rows = np.random.choice(len(self.memories), int(len(self.memories) * 0.9))
        self.memories = self.memories[rows, :]

        # 4. Put everything back together
        self.memories = self.memories.tolist() + recent_memories + important_memories

    # Exploration decay and plotting
    def finish_iteration(self, iteration):
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)

        self.reward_list.append(self.total_reward)
        self.total_reward = 0

        if len(self.reward_list) > 25:
            self.average_reward_list.append(np.mean(self.reward_list[:-25]))
        else:
            self.average_reward_list.append(np.mean(self.reward_list))

        if (iteration + 1) % self.plotting_iterations == 0:
            self.plot(iteration + 1)

    # Turn the pictures into a gif
    def make_animations(self):
        with imageio.get_writer("agent_learning.gif", mode="I") as writer:
            for file_name in self.file_names:
                image = imageio.imread(file_name)
                writer.append_data(image)

    # Currently in construction; check back later :)
    def plot(self, iteration):
        """
        fig = plt.figure(figsize=(20, 4), facecolor="white")
        fig.subplots_adjust(wspace=1)
        fig.suptitle(f"Iteration {iteration}")

        quality_table = self._generate_quality_table()

        quality_left = fig.add_subplot(1, 5, 1)
        quality_left.imshow(quality_table[:, :, 0].T, cmap="Spectral")
        quality_left.set_title("Quality for moving left")
        quality_left.set_xticks([])
        quality_left.set_yticks([])
        quality_left.set_xlabel("velocity")
        quality_left.set_ylabel("position")

        quality_left = fig.add_subplot(1, 5, 2)
        quality_left.imshow(quality_table[:, :, 1].T, cmap="Spectral")
        quality_left.set_title("Quality for moving right")
        quality_left.set_xticks([])
        quality_left.set_yticks([])
        quality_left.set_xlabel("velocity")
        quality_left.set_ylabel("position")

        policy = fig.add_subplot(1, 5, 4)
        policy.imshow(np.argmax(quality_table, axis=2).T, cmap="Spectral")
        policy.set_title("Policy: r=left, w=neutral, b=right")
        policy.set_xticks([])
        policy.set_yticks([])
        policy.set_xlabel("velocity")
        policy.set_ylabel("position")

        reward = fig.add_subplot(1, 5, 5)
        reward.plot(np.arange(len(self.reward_list)), self.reward_list, c="k")
        reward.plot(
            np.arange(len(self.reward_list)),
            self._average_reward_list,
            c="r",
            linewidth=2,
        )
        reward.set_title("Rewards over time")
        reward.set_xlabel("iterations")
        reward.set_ylabel("reward")
        reward.set_ylim([-200, 0])

        file_name = f"{self._image_path}/{iteration}.png"
        self._file_names.append(file_name)
        plt.savefig(file_name)
        plt.close("all")
        """
        return 0
