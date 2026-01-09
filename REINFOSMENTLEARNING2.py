import numpy as np
import matplotlib.pyplot as plt

states = 10
Q = np.zeros((states, 2))  # left/right

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 400

total_rewards = []

for ep in range(episodes):
    state = 0
    ep_reward = 0

    while state != states - 1:
        if np.random.rand() < epsilon:
            action = np.random.randint(2)
        else:
            action = np.argmax(Q[state])

        next_state = state + 1 if action == 1 else max(0, state - 1)

        if next_state == 5:      # cliff
            reward = -100
            next_state = 0
        elif next_state == states - 1:
            reward = 10
        else:
            reward = -1

        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state
        ep_reward += reward

    total_rewards.append(ep_reward)

plt.plot(total_rewards)
plt.xlabel("Episodes")
plt.ylabel("Episode Reward")
plt.title("Cliff Walking RL")
plt.show()
