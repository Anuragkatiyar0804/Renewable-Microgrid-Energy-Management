"""
make_graphs.py - project graphs, saved as PNG files in the results folder.
Graph 1: solar over the year (monthly)
Graph 2: household load over the year (monthly)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

SOLAR_FILE = "lucknow_solar_2024.csv"
LOAD_FILE = "household_load_all.csv"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def read_solar():
    with open(SOLAR_FILE) as f:
        lines = f.read().splitlines()
    start = [i for i, l in enumerate(lines) if l.startswith("YEAR,MO,DY,HR")][0]
    df = pd.read_csv(SOLAR_FILE, skiprows=start)
    return df[df["ALLSKY_SFC_SW_DWN"] >= 0]          # drop -999 missing values


def read_load():
    df = pd.read_csv(LOAD_FILE, header=None).iloc[:, :27]
    df.columns = ["year", "month", "day"] + list(range(24))
    return df


def bar_chart(values, title, ylabel, color, filename):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(MONTHS, values, color=color)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"results/{filename}", dpi=150)
    print(f"Saved results/{filename}")


def graph_solar():
    df = read_solar()
    daily = df.groupby(["MO", "DY"])["ALLSKY_SFC_SW_DWN"].sum() / 1000.0   # kWh/m2 per day
    monthly = daily.groupby(level=0).mean().reindex(range(1, 13))
    print("Solar, average kWh/m2 per day:\n", monthly.round(2).to_string(), "\n")
    bar_chart(monthly.values, "Solar energy over the year, Lucknow 2024 (monthly average)",
              "Solar irradiance (kWh/m² per day)", "gold", "solar_monthly.png")


def graph_load():
    df = read_load()
    print("Load file shape:", df.shape)
    daily_kwh = df[list(range(24))].sum(axis=1) / 1000.0
    month = df["month"].astype(int)
    monthly = daily_kwh.groupby(month).mean().reindex(range(1, 13))
    print("Days of load data per month:\n", month.value_counts().sort_index().to_string())
    print("Load, average kWh per day:\n", monthly.round(2).to_string(), "\n")
    bar_chart(monthly.values, "Household load over the year ","Load (kWh per day)", "tab:blue", "load_monthly.png")


def graph_avg_day():
    solar_hr = read_solar().groupby("HR")["ALLSKY_SFC_SW_DWN"].mean()
    load_hr = read_load()[list(range(24))].mean()

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    l1, = ax1.plot(solar_hr.index, solar_hr.values, color="goldenrod",
                   linewidth=2.5, label="Solar irradiance (left axis)")
    ax1.fill_between(solar_hr.index, solar_hr.values, color="gold", alpha=0.3)
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Solar irradiance (W/m²)")
    ax1.set_xlim(0, 23)
    ax1.set_xticks(range(0, 24, 3))
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    l2, = ax2.plot(range(24), load_hr.values, color="tab:blue",
                   linewidth=2.5, label="Household load (right axis)")
    ax2.set_ylabel("Household load (W)")

    ax1.set_title("Average day: solar vs household load by hour")
    ax1.legend(handles=[l1, l2], loc="upper left")
    fig.tight_layout()
    fig.savefig("results/avg_day.png", dpi=150)
    print("Saved results/avg_day.png")


import numpy as np
from dqn_agent import DQNAgent
from rule_based_ems import rule_based_action
from train_eval_year import build_pools, make_pairs, SEEDS
from train_eval_tod import TodEnv, eval_days as tod_eval, INR_PER_UNIT
from train_eval_tod_forecast import eval_days as fc_eval

DAYS_IN_MONTH = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]   # 2024 is a leap year


def load_agent(path, state_dim):
    agent = DQNAgent(state_dim=state_dim, action_dim=3, lr=0.002, gamma=0.95,
                     batch_size=32, target_update_freq=1000, eps_decay_steps=5000)
    agent.load(path)
    return lambda s: agent.select_action(s, greedy=True)


def get_results():
    """Cost of every test day for: rule, DQN (5 seeds), DQN with forecast (5 seeds)."""
    solar_pool, load_pool = build_pools()
    test_pairs = make_pairs(solar_pool["test"], load_pool["test"], 2)
    months = []
    for m in sorted(solar_pool["test"]):
        if m in load_pool["test"]:
            months += [m] * len(solar_pool["test"][m])
    rule = np.array(tod_eval(test_pairs, rule_based_action))
    dqn_seeds = np.array([tod_eval(test_pairs, load_agent(f"models/dqn_tod_seed{s}.pt", 6))
                          for s in SEEDS])
    fc_seeds = np.array([fc_eval(test_pairs, load_agent(f"models/dqn_forecast_seed{s}.pt", 8))
                         for s in SEEDS])
    return test_pairs, np.array(months), rule, dqn_seeds, fc_seeds


def graph_cost_comparison(rule, dqn_seeds, fc_seeds):
    vals = [rule.mean(), dqn_seeds.mean(), fc_seeds.mean()]
    errs = [0, dqn_seeds.mean(axis=1).std(), fc_seeds.mean(axis=1).std()]
    labels = ["Original\nrule-based", "DQN", "DQN with\nforecast"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(labels, vals, yerr=errs, capsize=6,
                  color=["tab:gray", "tab:blue", "tab:green"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"Rs {v:.2f}",
                ha="center", fontweight="bold")
    ax.set_ylabel("Average grid cost (Rs per day)")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title(f"Daily grid cost on {len(rule)} unseen test days (Rs 5 off-peak / Rs 8 peak)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/cost_comparison.png", dpi=150)
    print("Saved results/cost_comparison.png")


def graph_savings(months, rule, dqn_seeds, fc_seeds):
    dqn, fc = dqn_seeds.mean(axis=0), fc_seeds.mean(axis=0)
    dqn_sav, fc_sav = [], []
    for m in range(1, 13):
        mask = months == m
        dqn_sav.append((rule[mask] - dqn[mask]).mean() * DAYS_IN_MONTH[m - 1])
        fc_sav.append((rule[mask] - fc[mask]).mean() * DAYS_IN_MONTH[m - 1])
    x, w = np.arange(12), 0.4
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w / 2, dqn_sav, w, color="tab:blue",
                label=f"DQN (year total: Rs {sum(dqn_sav):,.0f})")
    b2 = ax.bar(x + w / 2, fc_sav, w, color="tab:green",
                label=f"DQN with forecast (year total: Rs {sum(fc_sav):,.0f})")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.0f}", ha="center",
                    va="bottom" if h >= 0 else "top", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS)
    ax.set_ylabel("Saving vs original rule-based (Rs per month)")
    ax.set_title("Estimated monthly and yearly saving in rupees")
    ax.set_xlabel("Estimate: average saving on the unseen test days of each month x days in the month")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(dqn_sav + fc_sav) * 1.3)
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig("results/monthly_saving.png", dpi=150)
    print("Saved results/monthly_saving.png")
    print(f"Yearly saving: DQN Rs {sum(dqn_sav):,.0f} | DQN with forecast Rs {sum(fc_sav):,.0f}")


def run_day(pair, policy):
    env = TodEnv({}, {}, seed=0, fixed=pair)
    state = env.reset()
    soc, cost, done = [env.battery_soc * 100], 0.0, False
    while not done:
        state, _, done, info = env.step(policy(state))
        soc.append(info["soc"] * 100)
        cost += info["grid_cost"] * INR_PER_UNIT
    return soc, cost


def graph_sample_day(test_pairs, rule, dqn_seeds):
    dqn_policy = load_agent("models/dqn_tod_seed0.pt", 6)
    idx = int(np.argsort(rule - dqn_seeds[0])[len(rule) // 2])     # median-saving day
    rule_soc, rule_cost = run_day(test_pairs[idx], rule_based_action)
    dqn_soc, dqn_cost = run_day(test_pairs[idx], dqn_policy)
    print(f"Sample day: test day {idx + 1} of {len(rule)} | rule Rs {rule_cost:.2f} | DQN Rs {dqn_cost:.2f}")

    hours = list(range(len(rule_soc)))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axvspan(9, 21, color="orange", alpha=0.12, label="Peak hours (Rs 8)")
    ax.plot(hours, rule_soc, color="tab:gray", linewidth=2,
            label=f"Original rule (day cost Rs {rule_cost:.2f})")
    ax.plot(hours, dqn_soc, color="tab:blue", linewidth=2,
            label=f"DQN (day cost Rs {dqn_cost:.2f})")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Battery charge (%)")
    ax.set_title("Battery charge on one sample test day (median-saving day)")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig("results/sample_day_battery.png", dpi=150)
    print("Saved results/sample_day_battery.png")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    graph_solar()
    graph_load()
    graph_avg_day()
    test_pairs, months, rule, dqn_seeds, fc_seeds = get_results()
    print(f"\nRule Rs {rule.mean():.2f} | DQN Rs {dqn_seeds.mean():.2f} | "
          f"DQN with forecast Rs {fc_seeds.mean():.2f} per day\n")
    graph_cost_comparison(rule, dqn_seeds, fc_seeds)
    graph_savings(months, rule, dqn_seeds, fc_seeds)
    graph_sample_day(test_pairs, rule, dqn_seeds)
    plt.show()