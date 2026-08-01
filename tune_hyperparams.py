"""
Hyperparameter tuning for the custom PyTorch DQN agent.

Runs a small random search over the hyperparameters that matter most for
DQN stability/performance:
    - learning rate
    - hidden layer size (via QNetwork - see note below)
    - batch size
    - target network update frequency
    - epsilon decay length
    - gamma (discount factor)

For each candidate, trains for a short budget (enough to see a trend, not
full convergence) across a few episodes, then evaluates average reward on
fresh episodes. The best config is then used for the full training run in
train_agent.py.

NOTE: to tune hidden_dim too, add a `hidden_dim` param to DQNAgent.__init__
that passes through to QNetwork(state_dim, action_dim, hidden_dim). Left as
default 128 here to keep the search fast; feel free to extend.
"""

import itertools
import random
import numpy as np

from microgrid_env import MicrogridEnv
from dqn_agent import DQNAgent


SEARCH_SPACE = {
    "lr": [1e-4, 5e-4, 1e-3, 2e-3],
    "batch_size": [32, 64, 128],
    "target_update_freq": [200, 500, 1000],
    "eps_decay_steps": [2000, 5000, 10000],
    "gamma": [0.95, 0.99],
}

TUNE_STEPS = 5000        # short training budget per candidate
EVAL_EPISODES = 5        # episodes to average for scoring
N_CANDIDATES = 12        # number of random configs to try


def sample_config():
    return {k: random.choice(v) for k, v in SEARCH_SPACE.items()}


def train_and_score(config: dict, seed: int = 0):
    env = MicrogridEnv(seed=seed)
    agent = DQNAgent(
        state_dim=6,
        action_dim=3,
        lr=config["lr"],
        gamma=config["gamma"],
        batch_size=config["batch_size"],
        target_update_freq=config["target_update_freq"],
        eps_decay_steps=config["eps_decay_steps"],
    )

    obs = env.reset()
    for _ in range(TUNE_STEPS):
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        agent.store_transition(obs, action, reward, next_obs, float(done))
        agent.train_step()
        obs = next_obs
        if done:
            obs = env.reset()

    # evaluate greedily (epsilon=0) on fresh episodes
    eval_env = MicrogridEnv(seed=seed + 1000)
    total_rewards = []
    for _ in range(EVAL_EPISODES):
        obs = eval_env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += reward
        total_rewards.append(ep_reward)

    return float(np.mean(total_rewards))


def tune():
    results = []
    for i in range(N_CANDIDATES):
        config = sample_config()
        score = train_and_score(config, seed=i)
        results.append((score, config))
        print(f"[{i+1}/{N_CANDIDATES}] score={score:.4f} | config={config}")

    results.sort(key=lambda x: x[0], reverse=True)
    best_score, best_config = results[0]
    print("\n=== BEST CONFIG ===")
    print(f"Score: {best_score:.4f}")
    print(best_config)
    return best_config


if __name__ == "__main__":
    tune()
