"""
Fair test under the REAL flat UPPCL tariff (Rs 6/kWh).
DQN trains and is tested in a flat-price world. Two baselines:
  1) original rule EMS (as written, buys grid power to charge when SOC < 35%)
  2) fair baseline: always solar first, then discharge, never charge from grid
"""

import os
import numpy as np
import torch

from dqn_agent import DQNAgent
from rule_based_ems import rule_based_action
from train_eval_year import (RealEnv, build_pools, make_pairs,
                             FLAT_RATE_INR, TRAIN_STEPS, EVAL_EVERY, SEEDS)


class FlatEnv(RealEnv):
    TRAIN_PRICE = 0.175   # same average price as the old peak/off-peak mix, keeps reward scale

    def __init__(self, solar_pool, load_pool, seed=None, fixed=None):
        super().__init__(solar_pool, load_pool, seed=seed, fixed=fixed)
        self.price_offpeak = self.TRAIN_PRICE
        self.price_peak = self.TRAIN_PRICE + 1e-6

    def _price(self, hour):
        return self.price_offpeak       # price never changes with the hour


def always_discharge(state):
    return 1                            # solar first, then battery, never grid-charge


def day_cost(env, policy):
    state = env.reset()
    grid_wh, done = 0.0, False
    while not done:
        state, _, done, info = env.step(policy(state))
        grid_wh += info["grid_w"]
    return grid_wh / 1000.0 * FLAT_RATE_INR


def eval_days(pairs, policy):
    return [day_cost(FlatEnv({}, {}, seed=0, fixed=p), policy) for p in pairs]


def main():
    os.makedirs("models", exist_ok=True)
    solar_pool, load_pool = build_pools()
    val_pairs = make_pairs(solar_pool["val"], load_pool["val"], 1)
    test_pairs = make_pairs(solar_pool["test"], load_pool["test"], 2)
    print(f"Validation days: {len(val_pairs)} | Test days: {len(test_pairs)} (never used in training)\n")

    orig_costs = eval_days(test_pairs, rule_based_action)
    fair_costs = eval_days(test_pairs, always_discharge)
    orig_avg, fair_avg = float(np.mean(orig_costs)), float(np.mean(fair_costs))
    print(f"Original rule-based:      Rs {orig_avg:.2f} per day")
    print(f"Fair baseline (no grid charging): Rs {fair_avg:.2f} per day\n")

    seed_avgs, wins_fair = [], []
    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = FlatEnv(solar_pool["train"], load_pool["train"], seed=seed)
        agent = DQNAgent(state_dim=6, action_dim=3, lr=0.002, gamma=0.95,
                         batch_size=32, target_update_freq=1000, eps_decay_steps=5000)
        greedy = lambda s: agent.select_action(s, greedy=True)
        best_val, best_path = float("inf"), f"models/dqn_flat_seed{seed}.pt"

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
        dqn_costs = eval_days(test_pairs, greedy)
        avg = float(np.mean(dqn_costs))
        seed_avgs.append(avg)
        wins_fair.append(sum(d < f - 1e-9 for d, f in zip(dqn_costs, fair_costs)))
        print(f"Seed {seed}: DQN Rs {avg:.2f}/day | vs original {(orig_avg - avg) / orig_avg * 100:+.1f}% "
              f"| vs fair {(fair_avg - avg) / fair_avg * 100:+.1f}%")

    m, s = np.mean(seed_avgs), np.std(seed_avgs)
    print("\n" + "=" * 55)
    print(f"Original rule-based:  Rs {orig_avg:.2f} per day")
    print(f"Fair baseline:        Rs {fair_avg:.2f} per day")
    print(f"DQN (5 seeds):        Rs {m:.2f} +/- {s:.2f} per day")
    print(f"DQN vs original rule: {(orig_avg - m) / orig_avg * 100:+.1f}%")
    print(f"DQN vs fair baseline: {(fair_avg - m) / fair_avg * 100:+.1f}%")
    print(f"DQN cheaper than fair baseline on avg {np.mean(wins_fair):.0f}/{len(test_pairs)} test days")


if __name__ == "__main__":
    main()