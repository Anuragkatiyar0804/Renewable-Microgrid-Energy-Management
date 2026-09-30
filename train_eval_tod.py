"""
What-if test: time-of-day tariff (Rs 5 off-peak, Rs 8 peak 09:00-21:00).
NOT the current UPPCL domestic tariff. Shows when RL should help.
Baselines: original rule EMS (designed for time-of-day prices) and always-discharge.
"""

import os
import numpy as np
import torch

from dqn_agent import DQNAgent
from rule_based_ems import rule_based_action
from train_eval_year import (RealEnv, build_pools, make_pairs,
                             TRAIN_STEPS, EVAL_EVERY, SEEDS)

INR_PER_UNIT = 50.0     # env price 0.10 -> Rs 5, 0.16 -> Rs 8


class TodEnv(RealEnv):
    def __init__(self, solar_pool, load_pool, seed=None, fixed=None):
        super().__init__(solar_pool, load_pool, seed=seed, fixed=fixed)
        self.price_peak = 0.16
        self.price_offpeak = 0.10


def always_discharge(state):
    return 1


def day_cost(env, policy):
    state = env.reset()
    cost, done = 0.0, False
    while not done:
        state, _, done, info = env.step(policy(state))
        cost += info["grid_cost"] * INR_PER_UNIT
    return cost


def eval_days(pairs, policy):
    return [day_cost(TodEnv({}, {}, seed=0, fixed=p), policy) for p in pairs]


def main():
    os.makedirs("models", exist_ok=True)
    solar_pool, load_pool = build_pools()
    val_pairs = make_pairs(solar_pool["val"], load_pool["val"], 1)
    test_pairs = make_pairs(solar_pool["test"], load_pool["test"], 2)
    print(f"Validation days: {len(val_pairs)} | Test days: {len(test_pairs)}\n")

    orig_avg = float(np.mean(eval_days(test_pairs, rule_based_action)))
    fair_avg = float(np.mean(eval_days(test_pairs, always_discharge)))
    print(f"Original rule (time-of-day aware): Rs {orig_avg:.2f} per day")
    print(f"Always-discharge baseline:         Rs {fair_avg:.2f} per day\n")

    seed_avgs = []
    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = TodEnv(solar_pool["train"], load_pool["train"], seed=seed)
        agent = DQNAgent(state_dim=6, action_dim=3, lr=0.002, gamma=0.95,
                         batch_size=32, target_update_freq=1000, eps_decay_steps=5000)
        greedy = lambda s: agent.select_action(s, greedy=True)
        best_val, best_path = float("inf"), f"models/dqn_tod_seed{seed}.pt"

        obs = env.reset()
        for step in range(1, TRAIN_STEPS + 1):
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, float(done))
            agent.train_step()
            obs = env.reset() if done else next_obs
            if step % EVAL_EVERY == 0:
                v = float(np.mean(eval_days(val_pairs, greedy)))
                if v < best_val:
                    best_val = v
                    agent.save(best_path)

        agent.load(best_path)
        avg = float(np.mean(eval_days(test_pairs, greedy)))
        seed_avgs.append(avg)
        print(f"Seed {seed}: DQN Rs {avg:.2f}/day | vs rule {(orig_avg - avg) / orig_avg * 100:+.1f}% "
              f"| vs always-discharge {(fair_avg - avg) / fair_avg * 100:+.1f}%")

    m, s = np.mean(seed_avgs), np.std(seed_avgs)
    print("\n" + "=" * 55)
    print(f"Original rule:      Rs {orig_avg:.2f} per day")
    print(f"Always-discharge:   Rs {fair_avg:.2f} per day")
    print(f"DQN (5 seeds):      Rs {m:.2f} +/- {s:.2f} per day")
    print(f"DQN vs rule:        {(orig_avg - m) / orig_avg * 100:+.1f}%")
    print(f"DQN vs always-discharge: {(fair_avg - m) / fair_avg * 100:+.1f}%")


if __name__ == "__main__":
    main()