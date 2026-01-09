import numpy as np
import matplotlib.pyplot as plt

# Environment
n_states = 5
n_actions = 2  # 0 = left, 1 = right
goal_state = 4

# Q-table
Q = np.zeros((n_states, n_actions))

# Hyperparameters
alpha = 0.1      # learning rate
gamma = 0.9      # discount factor
epsilon = 0.2    # exploration rate
episodes = 200

steps_per_episode = []

# Training loop
for episode in range(episodes):
    state = 0  # start position
    steps = 0

    while state != goal_state:
        # Choose action (epsilon-greedy)
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(Q[state])

        # Take action
        if action == 0:  # move left
            next_state = max(0, state - 1)
        else:           # move right
            next_state = min(goal_state, state + 1)

        # Reward
        reward = 10 if next_state == goal_state else -1

        # Q-learning update
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state
        steps += 1

    steps_per_episode.append(steps)

# Plot learning curve
plt.plot(steps_per_episode)
plt.xlabel("Episodes")
plt.ylabel("Steps to reach goal")
plt.title("Reinforcement Learning: Q-Learning Progress")
plt.show()
