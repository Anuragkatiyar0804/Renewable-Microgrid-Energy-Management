"""
Full-year real-data test: proper train / validation / test split, 5 random seeds.
Solar: NASA POWER Lucknow 2024. Load: UCI household (2006-2010), same calendar month.
Split by day of month: 1-21 train, 22-25 validation, 26-31 test.
Billed at flat Rs 6/kWh.
"""

import os
import numpy as np
import pandas as pd
import torch

from microgrid_env import MicrogridEnv
from dqn_agent import DQNAgent
from rule_based_ems import rule_based_action

PERFORMANCE_RATIO = 0.8
FLAT_RATE_INR = 6.00
TRAIN_STEPS = 60_000
EVAL_EVERY = 10_000
SEEDS = [0, 1, 2, 3, 4]


def split_of(dom):
    return "train" if dom <= 21 else ("val" if dom <= 25 else "test")


class RealEnv(MicrogridEnv):
    def __init__(self, solar_pool, load_pool, seed=None, fixed=None):
        super().__init__(seed=seed)
        self.solar_pool = solar_pool
        self.load_pool = load_pool
        self.fixed = fixed
        self.months = [m for m in solar_pool if m in load_pool]
        self.solar_day = None
        self.load_day = None

    def reset(self):
        if self.fixed is not None:
            self.solar_day, self.load_day = self.fixed
        else:
            m = self.months[int(self.rng.integers(len(self.months)))]
            s, l = self.solar_pool[m], self.load_pool[m]
            self.solar_day = s[int(self.rng.integers(len(s)))]
            self.load_day = l[int(self.rng.integers(len(l)))]
        return super().reset()

    def _solar_output_w(self, hour):
        irr = self.solar_day[min(int(hour), 23)]
        return max(irr, 0.0) / 1000.0 * self.solar_peak_w * PERFORMANCE_RATIO

    def _load_w(self, hour):
        w = self.load_day[min(int(hour), 23)]
        return float(np.clip(w, self.load_min_w, self.load_max_w))


def build_pools():
    sol = pd.read_csv("lucknow_solar_2024.csv", skiprows=9)
    irr = sol["ALLSKY_SFC_SW_DWN"].clip(lower=0).values.reshape(-1, 24)
    s_month, s_dom = sol["MO"].values[::24], sol["DY"].values[::24]

    load = np.loadtxt("household_load_all.csv", delimiter=",")
    l_month, l_dom, l_days = load[:, 1].astype(int), load[:, 2].astype(int), load[:, 3:]

    def pool(days, months, doms):
        splits = np.array([split_of(d) for d in doms])
        p = {k: {} for k in ("train", "val", "test")}
        for k in p:
            for m in range(1, 13):
                mask = (months == m) & (splits == k)
                if mask.any():
                    p[k][m] = days[mask]
        return p

    return pool(irr, s_month, s_dom), pool(l_days, l_month, l_dom)


def make_pairs(solar_p, load_p, seed):
    rng = np.random.default_rng(seed)
    pairs = []
    for m in sorted(solar_p):
        if m not in load_p:
            continue
        for s in solar_p[m]:
            pairs.append((s, load_p[m][int(rng.integers(len(load_p[m])))]))
    return pairs


def day_cost(env, policy):
    state = env.reset()
    grid_wh, done = 0.0, False
    while not done:
        state, _, done, info = env.step(policy(state))
        grid_wh += info["grid_w"]          # 1-hour steps, so W = Wh
    return grid_wh / 1000.0 * FLAT_RATE_INR


def eval_days(pairs, policy):
    return [day_cost(RealEnv({}, {}, seed=0, fixed=p), policy) for p in pairs]


def main():
    os.makedirs("models", exist_ok=True)
    solar_pool, load_pool = build_pools()
    val_pairs = make_pairs(solar_pool["val"], load_pool["val"], 1)
    test_pairs = make_pairs(solar_pool["test"], load_pool["test"], 2)
    print(f"Validation days: {len(val_pairs)} | Test days: {len(test_pairs)} (never used in training)\n")

    rule_costs = eval_days(test_pairs, rule_based_action)
    rule_avg = float(np.mean(rule_costs))
    print(f"Rule-based on test days: Rs {rule_avg:.2f} per day\n")

    seed_avgs, wins, total = [], 0, 0
    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = RealEnv(solar_pool["train"], load_pool["train"], seed=seed)
        agent = DQNAgent(state_dim=6, action_dim=3, lr=0.002, gamma=0.95,
                         batch_size=32, target_update_freq=1000, eps_decay_steps=5000)
        greedy = lambda s: agent.select_action(s, greedy=True)
        best_val, best_path = float("inf"), f"models/dqn_year_seed{seed}.pt"

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
        wins += sum(d < r for d, r in zip(dqn_costs, rule_costs))
        total += len(dqn_costs)
        print(f"Seed {seed}: DQN test cost Rs {avg:.2f}/day | "
              f"vs rule-based {(rule_avg - avg) / rule_avg * 100:+.1f}%")

    m, s = np.mean(seed_avgs), np.std(seed_avgs)
    print("\n" + "=" * 50)
    print(f"Rule-based:  Rs {rule_avg:.2f} per day")
    print(f"DQN (5 seeds): Rs {m:.2f} +/- {s:.2f} per day")
    print(f"Improvement: {(rule_avg - m) / rule_avg * 100:+.1f}%")
    print(f"DQN cheaper on {wins}/{total} test-day runs ({wins / total * 100:.0f}%)")


if __name__ == "__main__":
    main()