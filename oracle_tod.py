"""
Best-possible cost per day (perfect foresight) under the time-of-day tariff.
Dynamic programming over battery charge. It is an optimistic ceiling on savings,
because a real controller cannot know the day's solar and load in advance.
"""

import numpy as np

from train_eval_year import build_pools, make_pairs
from train_eval_tod import TodEnv, always_discharge, eval_days, INR_PER_UNIT

N_SOC = 61
GRID = np.linspace(0.0, 1.0, N_SOC)


def oracle_cost(pair):
    env = TodEnv({}, {}, seed=0, fixed=pair)
    env.reset()
    V = np.zeros(N_SOC)                 # cost after hour 24 is zero
    for h in range(23, -1, -1):
        newV = np.zeros(N_SOC)
        for i, soc in enumerate(GRID):
            best = float("inf")
            for a in (0, 1, 2):
                env.hour = h
                env.battery_soc = float(soc)
                env._get_state()        # sets this hour's solar, load and price
                _, _, _, info = env.step(a)
                nxt = np.interp(info["soc"], GRID, V)
                best = min(best, info["grid_cost"] * INR_PER_UNIT + nxt)
            newV[i] = best
        V = newV
    return float(np.interp(0.5, GRID, V))   # every day starts at 50% charge


def main():
    solar_pool, load_pool = build_pools()
    test_pairs = make_pairs(solar_pool["test"], load_pool["test"], 2)
    print(f"Computing best possible cost on {len(test_pairs)} test days...\n")

    best = float(np.mean([oracle_cost(p) for p in test_pairs]))
    fair = float(np.mean(eval_days(test_pairs, always_discharge)))

    print(f"Always-discharge:              Rs {fair:.2f} per day")
    print(f"Best possible (perfect foresight): Rs {best:.2f} per day")
    print(f"Room to save: {(fair - best) / fair * 100:.1f}%")


if __name__ == "__main__":
    main()