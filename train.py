import os
import numpy as np
from simulation.rl_env import TrafficEnv
from simulation.dqn_agent import DQNAgent


def train_dqn(episodes: int = 30, batch_size: int = 32, save_path: str = "model.pth"):
    print("=" * 60)
    print("      DEEP Q-NETWORK (DQN) TRAFFIC AGENT TRAINING      ")
    print("=" * 60)

    env = TrafficEnv(target_intersection_id="I_1_1", episode_length=100, spawn_rate=0.4, seed=42)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=1e-3, gamma=0.95, epsilon=1.0)

    rewards_history = []
    wait_times_history = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        losses = []

        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay(batch_size=batch_size)
            if loss is not None:
                losses.append(loss)

            state = next_state
            episode_reward += reward

            if done:
                avg_wait = info["avg_wait_time"]
                break

        # Periodically update target Q-network
        if ep % 5 == 0:
            agent.update_target_network()

        mean_loss = np.mean(losses) if losses else 0.0
        rewards_history.append(episode_reward)
        wait_times_history.append(avg_wait)

        print(
            f"[Episode {ep:02d}/{episodes:02d}] "
            f"Reward: {episode_reward:8.2f} | "
            f"Avg Wait: {avg_wait:5.2f}s | "
            f"Epsilon: {agent.epsilon:5.3f} | "
            f"Mean Loss: {mean_loss:6.4f}"
        )

    # Save trained PyTorch model weights
    agent.save(save_path)
    print("=" * 60)
    print(f"Training Complete! Model weights saved successfully to '{save_path}'.")
    print(f"Initial Ep 1 Avg Wait: {wait_times_history[0]}s -> Final Ep {episodes} Avg Wait: {wait_times_history[-1]}s")
    print("=" * 60)

    return agent, rewards_history, wait_times_history


if __name__ == "__main__":
    train_dqn(episodes=30)
