"""
Train the custom PyTorch DQN agent on the microgrid environment.

This replaces the Stable-Baselines3 `model.learn()` call with a manual
training loop, so you control and can explain every step: action selection,
storing transitions, sampling batches, computing loss, updating networks.

Assumes `microgrid_env.py` exposes a Gym-style `MicrogridEnv` class with:
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    env.observation_space.shape[0]  -> state_dim (6 in the reference repo)
    env.action_space.n              -> action_dim (3 in the reference repo)

If your env.py differs slightly (e.g. reset() returns (obs, info) as in
newer gymnasium versions), adjust the two spots marked "ADAPT HERE".
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from dqn_agent import DQNAgent
from microgrid_env import MicrogridEnv  # your own env file


def train(algo_steps: int, save_dir: str = "models"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    env = MicrogridEnv(seed=42)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=0.002,
        gamma=0.95,
        batch_size=32,
        target_update_freq=1000,
        eps_decay_steps=5000,
    )

    episode_rewards = []
    losses = []

    obs = env.reset()

    episode_reward = 0.0
    episode = 0

    for step in range(1, algo_steps + 1):
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)

        agent.store_transition(obs, action, reward, next_obs, float(done))
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)

        obs = next_obs
        episode_reward += reward

        if done:
            episode += 1
            episode_rewards.append(episode_reward)
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                eps = agent.current_epsilon()
                print(f"Episode {episode} | Step {step}/{algo_steps} | "
                      f"AvgReward(10): {avg_reward:.3f} | Epsilon: {eps:.3f}")
            obs = env.reset()
            episode_reward = 0.0

    agent.save(os.path.join(save_dir, "dqn_microgrid.pt"))

    # Training curve plot (mirrors the repo's dqn_training_curve.png)
    plt.figure(figsize=(8, 5))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Curve (Custom PyTorch)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dqn_training_curve.png"))
    print(f"Saved model to {save_dir}/dqn_microgrid.pt")
    print(f"Saved training curve to {save_dir}/dqn_training_curve.png")

    return agent, episode_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000,
                         help="Total environment steps to train for")
    parser.add_argument("--quick", action="store_true",
                         help="Quick 10k-step smoke test")
    args = parser.parse_args()

    total_steps = 10_000 if args.quick else args.steps
    train(total_steps)
