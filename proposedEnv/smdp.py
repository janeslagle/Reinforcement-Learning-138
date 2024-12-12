import numpy as np

class SMDP:
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.1, max_steps=1000):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.q_table = np.zeros((env.state_size, len(env.action_space)))
        self.action_history = []

    def choose_action(self, state):
        state_index = int(np.mean(state) / 10)
        state_index = np.clip(state_index, 0, self.q_table.shape[0] - 1)

        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.env.action_space)
        else:
            action = self.env.action_space[np.argmax(self.q_table[state_index])]

        self.action_history.append(action)
        return action

    def update_q_table(self, state, action, reward, next_state, done, dt):
        state_index = int(np.mean(state) / 10)
        next_state_index = int(np.mean(next_state) / 10)

        state_index = np.clip(state_index, 0, self.q_table.shape[0] - 1)
        next_state_index = np.clip(next_state_index, 0, self.q_table.shape[0] - 1)

        action_index = self.env.action_space.index(action)

        next_max = np.max(self.q_table[next_state_index])
        self.q_table[state_index, action_index] += self.alpha * (
            reward + self.gamma * next_max * dt - self.q_table[state_index, action_index]
        )

    def train(self, episodes=100):
        rewards = []
        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                dt = np.random.randint(1, 5) if self.env.is_smdp else 1
                next_state, reward, done, _ = self.env.step(action, dt)
                self.update_q_table(state, action, reward, next_state, done, dt)
                state = next_state
                total_reward += reward

            rewards.append(total_reward)
            if ep % 10 == 0:
                print(f"Episode {ep}: Total Reward = {total_reward}")
        return rewards
