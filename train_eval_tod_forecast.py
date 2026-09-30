"""
Time-of-day what-if (Rs 5 off-peak / Rs 8 peak) with a same-day FORECAST given to the DQN.
The DQN sees 2 extra inputs: expected solar and expected load for the rest of the day
(with forecast error). The baselines see the normal 6 inputs.
"""

import os
import numpy as np
import torch

from dqn_agent import DQNAgent
from rule_based_ems import rule_based_action
from train_eval_year import build_pools, make_pairs, TRAIN_STEPS, EVAL_EVERY, SEEDS
from train_eval_tod import TodEnv, INR_PER_UNIT

FORECAST_ERROR = 0.15        # forecast is off by about 15% (edit if you want)
ORACLE_RS = 10.81            # best possible cost from oracle_tod.py


def _rng_for(arr):
    return np.random.default_rng(int(abs(arr.sum()) * 1000) % (2 ** 32))


class ForecastEnv(TodEnv):
    def __init__(self, solar_pool, load_pool, seed=None, fixed=None):
        super().__init__(solar_pool, load_pool, seed=seed, fixed=fixed)
        self.fc_solar = 1.0
        self.fc_load = 1.0

    def _get_state(self):
        if int(self.hour) == 0:          # new day: draw this day's forecast errors
            self.fc_solar = 1.0 + FORECAST_ERROR * _rng_for(self.solar_day).normal()
            self.fc_load = 1.0 + FORECAST_ERROR * _rng_for(self.load_day).normal()
        s = super()._get_state()
        h = int(self.hour)
        rem_solar = sum(self._solar_output_w(k) for k in range(h, 24)) * self.fc_solar / 6000.0
        rem_load = sum(self._load_w(k) for k in range(h, 24)) * self.fc_load / 6000.0
        return np.append(s, [rem_solar, rem_load]).astype(np.float32)

    def step(self, action):
        s, r, d, info = super().step(action)
        if d:
            s = np.zeros(8, dtype=np.float32)
        return s, r, d, info


def day_cost(env, policy):
    state = env.reset()
    cost, done = 0.0, False
    while not done:
        state, _, done, info = env.step(policy(state))
        cost += info["grid_cost"] * INR_PER_UNIT
    return cost


def eval_days(pairs, policy):
    return [day_cost(ForecastEnv({}, {}, seed=0, fixed=p), policy) for p in pairs]


def main():
    os.makedirs("models", exist_ok=True)
    solar_pool, load_pool = build_pools()
    val_pairs = make_pairs(solar_pool["val"], load_pool["val"], 1)
    test_pairs = make_pairs(solar_pool["test"], load_pool["test"], 2)
    print(f"Validation days: {len(val_pairs)} | Test days: {len(test_pairs)}\n")

    orig = float(np.mean(eval_days(test_pairs, lambda s: rule_based_action(s[:6]))))
    fair = float(np.mean(eval_days(test_pairs, lambda s: 1)))
    print(f"Original rule:       Rs {orig:.2f} per day")
    print(f"Always-discharge:    Rs {fair:.2f} per day")
    print(f"Best possible:       Rs {ORACLE_RS:.2f} per day (perfect foresight)\n")

    seed_avgs = []
    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = ForecastEnv(solar_pool["train"], load_pool["train"], seed=seed)
        agent = DQNAgent(state_dim=8, action_dim=3, lr=0.002, gamma=0.95,
                         batch_size=32, target_update_freq=1000, eps_decay_steps=5000)
        greedy = lambda s: agent.select_action(s, greedy=True)
        best_val, best_path = float("inf"), f"models/dqn_forecast_seed{seed}.pt"

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
        print(f"Seed {seed}: DQN Rs {avg:.2f}/day | vs always-discharge {(fair - avg) / fair * 100:+.1f}%")

    m, s = np.mean(seed_avgs), np.std(seed_avgs)
    print("\n" + "=" * 55)
    print(f"Always-discharge:        Rs {fair:.2f} per day")
    print(f"DQN with forecast (5 seeds): Rs {m:.2f} +/- {s:.2f} per day")
    print(f"Best possible:           Rs {ORACLE_RS:.2f} per day")
    print(f"DQN vs always-discharge: {(fair - m) / fair * 100:+.1f}%")
    print(f"Share of possible savings captured: {(fair - m) / (fair - ORACLE_RS) * 100:.0f}%")


if __name__ == "__main__":
    main()

