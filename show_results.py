"""
Show the final results WITHOUT retraining.
Loads the saved models, runs them and the baselines on the same 66 unseen test days,
prints the table and saves results/real_data_comparison.png.
Needs: lucknow_solar_2024.csv, household_load_all.csv and the models/ folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from dqn_agent import DQNAgent
from rule_based_ems import rule_based_action
from train_eval_year import build_pools, make_pairs, SEEDS
from train_eval_flat import eval_days as flat_eval
from train_eval_tod import eval_days as tod_eval
from train_eval_tod_forecast import eval_days as fc_eval, ORACLE_RS


def dqn_result(eval_fn, pairs, prefix, state_dim):
    avgs = []
    for seed in SEEDS:
        agent = DQNAgent(state_dim=state_dim, action_dim=3)
        agent.load(f"models/{prefix}_seed{seed}.pt")
        costs = eval_fn(pairs, lambda s: agent.select_action(s, greedy=True))
        avgs.append(float(np.mean(costs)))
    return float(np.mean(avgs)), float(np.std(avgs))


def main():
    solar_pool, load_pool = build_pools()
    pairs = make_pairs(solar_pool["test"], load_pool["test"], 2)
    print(f"Testing on {len(pairs)} unseen days (real Lucknow solar + real household load)\n")

    rule = lambda s: rule_based_action(s[:6])
    always = lambda s: 1
    settings = [
        ("Flat Rs 6/kWh", flat_eval, "dqn_flat", 6),
        ("Peak/off-peak\nRs 5 / Rs 8", tod_eval, "dqn_tod", 6),
        ("Peak/off-peak\n+ forecast", fc_eval, "dqn_forecast", 8),
    ]

    rows = []
    for name, fn, prefix, dim in settings:
        orig = float(np.mean(fn(pairs, rule)))
        fair = float(np.mean(fn(pairs, always)))
        m, sd = dqn_result(fn, pairs, prefix, dim)
        rows.append((name, orig, fair, m, sd))

    print(f"{'Setting':<28}{'Original rule':>14}{'Always-discharge':>18}{'DQN (5 seeds)':>20}")
    print("-" * 80)
    for name, orig, fair, m, sd in rows:
        print(f"{name.replace(chr(10), ' '):<28}{orig:>14.2f}{fair:>18.2f}{m:>14.2f} +/- {sd:.2f}")
    print(f"\nBest possible (perfect foresight, peak/off-peak): Rs {ORACLE_RS:.2f} per day")

    x = np.arange(len(rows))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w, [r[1] for r in rows], w, label="Original rule", color="tab:orange")
    ax.bar(x, [r[2] for r in rows], w, label="Always-discharge", color="tab:gray")
    ax.bar(x + w, [r[3] for r in rows], w, yerr=[r[4] for r in rows], capsize=4,
           label="DQN (5 seeds)", color="tab:blue")
    ax.hlines(ORACLE_RS, 0.6, 2.4, colors="green", linestyles="--",
              label=f"Best possible, peak/off-peak (Rs {ORACLE_RS:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("Grid cost per day (Rs)")
    ax.set_title(f"Real-data test on {len(pairs)} unseen days")
    ax.set_ylim(0, 19)
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/real_data_comparison.png", dpi=120)
    print("Saved chart to results/real_data_comparison.png")


if __name__ == "__main__":
    main()