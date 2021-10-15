import os, shutil, imageio, random
import numpy as np
import matplotlib.pyplot as plt


class RandomAgent(object):
    def __init__(self, environment):
        self.environment = environment

    #  *_  eats any number of arguments
    def act(self, *_):
        return self.environment.action_space.sample()

    def convert_state(self, state):
        return state

    def learn(self, *_):
        return


# The base tabular agent class. This is unable to learn so the class must be extended
# and the learn function must be implemented.
class TabularAgent(RandomAgent):
    def __init__(self, environment):
        self.environment = environment
        self.action_space = environment.action_space.n
        self.observation_space = environment.observation_space.shape[0]
        self.state_space = 20

        # Learning updates
        self.learning_rate = 0.1
        self.discount = 0.95

        # Exploration
        self.min_exploration_rate = 0.01
        self.exploration_rate = 1.0
        self.exploration_decay = 0.9995

        # Initialize a table to hold an expected value for every state-action pair.
        self.quality_table = np.zeros(
            shape=(self.state_space, self.state_space, self.action_space)
        )

        # Memory replay
        self.max_memories = 5000
        self.memories = []

        # Plotting variables
        self.file_names = []
        self.trajectory = []
        self.reward_list = []
        self.average_reward_list = []
        self.total_reward = 0
        self.plotting_iterations = 250
        self.image_path = "./temp_images"
        if os.path.exists(self.image_path):
            shutil.rmtree(self.image_path)
        os.mkdir(self.image_path)

    # Explore early on, then exploit the learned knowledge
    # Actions:
    #   Type: Discrete(3)
    #   Num    Action
    #   0      Accelerate to the Left
    #   1      Don't accelerate
    #   2      Accelerate to the Right
    def act(self, state):
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_space)
        else:
            actions = self.quality_table[tuple(self.state_to_index(state))]
            return np.argmax(actions)

    # Converts the continuous values to integers for the table
    # Observation:
    #   Type: Box(2)
    #   Num    Observation               Min            Max
    #   0      Car Position              -1.2           0.6
    #   1      Car Velocity              -0.07          0.07
    def state_to_index(self, state):
        position, velocity = state.copy()

        new_min, new_max = +0, +self.state_space
        linear_scaling = lambda x, old_min, old_max: np.trunc(
            np.interp(x, (old_min, old_max), (new_min, new_max))
        )
        position = linear_scaling(
            position,
            self.environment.observation_space.low[0],
            self.environment.observation_space.high[0],
        )
        velocity = linear_scaling(
            velocity,
            self.environment.observation_space.low[1],
            self.environment.observation_space.high[1],
        )

        return int(position), int(velocity)

    # The learn function is unfinished, but contains plotting instructions
    def learn(self, state, next_state, action, reward, *_):
        self.trajectory.append(self.state_to_index(state))
        self.total_reward += reward

        if len(self.memories) > self.max_memories:
            self.clean_memories()

    # At the end of an iteration we want to decay the exploration rate,
    # get some plotting information, and make plots.
    def finish_iteration(self, iteration):
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)

        self.trajectory = np.array(self.trajectory)
        self.reward_list.append(self.total_reward)

        if len(self.reward_list) > 250:
            self.average_reward_list.append(np.mean(self.reward_list[:-250]))
        else:
            self.average_reward_list.append(np.mean(self.reward_list))

        if (iteration + 1) % self.plotting_iterations == 0:
            self.plot(iteration + 1)

        self.trajectory = []
        self.total_reward = 0

    def plot(self, iteration):
        fig = plt.figure(figsize=(20, 4), facecolor="white")
        fig.subplots_adjust(wspace=1)
        fig.suptitle(f"Iteration {iteration}")

        quality_left = fig.add_subplot(1, 5, 1)
        quality_left.imshow(self.quality_table[:, :, 0].T, cmap="Spectral")
        quality_left.set_title("Quality for moving left")
        quality_left.set_xticks([])
        quality_left.set_yticks([])
        quality_left.set_xlabel("velocity")
        quality_left.set_ylabel("position")

        quality_left = fig.add_subplot(1, 5, 2)
        quality_left.imshow(self.quality_table[:, :, 1].T, cmap="Spectral")
        quality_left.set_title("Quality for not moving")
        quality_left.set_xticks([])
        quality_left.set_yticks([])
        quality_left.set_xlabel("velocity")
        quality_left.set_ylabel("position")

        quality_left = fig.add_subplot(1, 5, 3)
        quality_left.imshow(self.quality_table[:, :, 2].T, cmap="Spectral")
        quality_left.set_title("Quality for moving right")
        quality_left.set_xticks([])
        quality_left.set_yticks([])
        quality_left.set_xlabel("velocity")
        quality_left.set_ylabel("position")

        policy = fig.add_subplot(1, 5, 4)
        policy.imshow(np.argmax(self.quality_table, axis=2).T, cmap="Spectral")
        policy.plot(
            self.trajectory[:, 0], self.trajectory[:, 1], c="k", linewidth=2,
        )
        policy.set_title("Policy: r=left, w=neutral, b=right")
        policy.set_xticks([])
        policy.set_yticks([])
        policy.set_xlabel("velocity")
        policy.set_ylabel("position")

        reward = fig.add_subplot(1, 5, 5)
        reward.plot(np.arange(len(self.reward_list)), self.reward_list, c="k")
        reward.plot(
            np.arange(len(self.reward_list)),
            self.average_reward_list,
            c="r",
            linewidth=2,
        )
        reward.set_title("Rewards over time")
        reward.set_xlabel("iterations")
        reward.set_ylabel("reward")
        reward.set_ylim([-200, 0])

        file_name = f"{self.image_path}/{iteration}.png"
        self.file_names.append(file_name)
        plt.savefig(file_name)
        plt.close("all")

    def make_animations(self):
        with imageio.get_writer("agent_learning.gif", mode="I") as writer:
            for file_name in self.file_names:
                image = imageio.imread(file_name)
                writer.append_data(image)

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


# Monte Carlo has a hard time with Mountain Car since an action in a state
# does not guarantee movement into another state
class TabularAgentMonteCarlo(TabularAgent):
    def __init__(self, environment):
        super().__init__(environment)

        self.actions = []
        self.states = []
        self.rewards = []

    def learn(self, state, next_state, action, reward, done):
        super().learn(state, next_state, action, reward, done)

        self.actions.append(action)  # [left, right, coast, ...] 200 long
        self.states.append(self.state_to_index(state))  # [state0, state1, ...]
        self.rewards.append(reward)  # [-1, -1, -1, -1, ..., +1] 200 long

        if done:
            for index in range(len(self.states)):
                action = self.actions[index]
                state = self.states[index]

                discounted_reward = 0
                for t in range(index, len(self.rewards)):
                    discounted_reward += (self.discount ** (t - index)) * self.rewards[
                        t
                    ]

                update = self.learning_rate * (
                    discounted_reward - self.quality_table[state + (action,)]
                )
                self.quality_table[state + (action,)] += update

    def finish_iteration(self, iteration):
        super().finish_iteration(iteration)

        self.actions = []
        self.rewards = []
        self.states = []


# On-policy Temporal Difference is also known as: SARSA
class TabularAgentOnPolicyTD(TabularAgent):
    def __init__(self, environment):
        super().__init__(environment)

    def learn(self, state, next_state, action, reward, done):
        super().learn(state, next_state, action, reward, done)

        next_action = self.act(next_state)
        state = self.state_to_index(state)
        next_state = self.state_to_index(next_state)

        self.td_update(state, next_state, action, reward, next_action)

    def td_update(self, state, next_state, action, reward, next_action):
        update = self.learning_rate * (
            reward
            + self.discount * self.quality_table[next_state + (next_action,)]
            - self.quality_table[state + (action,)]
        )
        self.quality_table[state + (action,)] += update


# Off-policy Temporal Difference is also known as: MAXSARSA and Q-Learning
class TabularAgentOffPolicyTD(TabularAgent):
    def __init__(self, environment):
        super().__init__(environment)

    def learn(self, state, next_state, action, reward, done):
        super().learn(state, next_state, action, reward, done)

        state = self.state_to_index(state)
        next_state = self.state_to_index(next_state)

        self.td_update(state, next_state, action, reward)

    def td_update(self, state, next_state, action, reward):
        update = self.learning_rate * (
            reward
            + self.discount * np.amax(self.quality_table[next_state])
            - self.quality_table[state + (action,)]
        )
        self.quality_table[state + (action,)] += update


class TabularAgentTDN(TabularAgent):
    def __init__(self, environment):
        super().__init__(environment)

        self.n = 3
        self.learning_rate = 0.05
        self.discount = 0.90

        self.actions = []
        self.states = []
        self.rewards = []

    def learn(self, state, next_state, action, reward, done):
        super().learn(state, next_state, action, reward, done)

        self.actions.append(action)
        self.states.append(self.state_to_index(state))
        self.rewards.append(reward)

        if len(self.actions) >= self.n:
            self.td_update()

        if done:
            for _ in range(self.n - 1):
                self.td_update()

    def finish_iteration(self, iteration):
        super().finish_iteration(iteration)

    def td_update(self):
        action = self.actions[0]
        state = self.states[0]
        discounted_reward = 0

        for index, reward in enumerate(self.rewards):
            discounted_reward += (self.discount ** index) * reward

        update = self.learning_rate * (
            discounted_reward - self.quality_table[state + (action,)]
        )
        self.quality_table[state + (action,)] += update

        self.actions = self.actions[1:]
        self.states = self.states[1:]
        self.rewards = self.rewards[1:]


class TabularAgentDynaQ(TabularAgent):
    def __init__(self, environment):
        super().__init__(environment)
        self.learning_rate = 0.01

        self.max_memories = 2000
        self.model = np.zeros(
            shape=(self.state_space, self.state_space, self.action_space), dtype=object
        )

    def learn(self, state, next_state, action, reward, done):
        super().learn(state, next_state, action, reward, done)

        state = self.state_to_index(state)
        next_state = self.state_to_index(next_state)

        self.td_update(state, next_state, action, reward)
        self.dyna_q_remember(state, next_state, action, reward)
        self.dyna_q_learn()

    def td_update(self, state, next_state, action, reward):
        update = self.learning_rate * (
            reward
            + self.discount * np.amax(self.quality_table[next_state])
            - self.quality_table[state + (action,)]
        )
        self.quality_table[state + (action,)] += update

    def dyna_q_remember(self, state, next_state, action, reward):
        self.memories.append([0, state, action])
        self.model[state + (action,)] = (reward, next_state)

    def dyna_q_learn(self):
        if len(self.memories) < self.max_memories:
            return
        for _, state, action in random.sample(self.memories, self.max_memories):
            reward, next_state = self.model[state + (action,)]
            self.td_update(state, next_state, action, reward)
