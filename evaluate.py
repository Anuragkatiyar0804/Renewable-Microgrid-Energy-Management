"""
Evaluate the trained DQN agent against the rule-based baseline EMS.

Produces:
    - printed comparison metrics (grid cost, solar self-consumption)
    - results/comparison_plot.png (cost + SOC trajectory for both policies)
"""

import numpy as np
import matplotlib.pyplot as plt

from microgrid_env import MicrogridEnv
from dqn_agent import DQNAgent
from rule_based_ems import rule_based_action


def run_episode(env, policy_fn):
    state = env.reset()
    total_cost = 0.0
    total_solar_used = 0.0
    total_solar_curtailed = 0.0
    total_pv_generated = 0.0
    soc_trace = [state[2]]
    cost_trace = []
    done = False
    while not done:
        action = policy_fn(state)
        state, reward, done, info = env.step(action)
        total_cost += info["grid_cost"]
        total_solar_used += info["solar_used_w"]
        total_solar_curtailed += info["solar_curtailed_w"]
        total_pv_generated += info["pv_w"]
        soc_trace.append(info["soc"])
        cost_trace.append(info["grid_cost"])
    utilization = total_solar_used / total_pv_generated if total_pv_generated > 0 else 0
    return total_cost, utilization, total_solar_curtailed, soc_trace, cost_trace


def main():
    N_EVAL_EPISODES = 20
    model_path = "models/dqn_microgrid.pt"

    dqn_agent = DQNAgent(state_dim=6, action_dim=3)
    dqn_agent.load(model_path)
    dqn_policy = lambda state: dqn_agent.select_action(state, greedy=True)

    dqn_costs, dqn_utils, dqn_curtails = [], [], []
    rule_costs, rule_utils, rule_curtails = [], [], []
    dqn_soc_trace, dqn_cost_trace = None, None
    rule_soc_trace, rule_cost_trace = None, None

    for ep in range(N_EVAL_EPISODES):
        env_dqn = MicrogridEnv(seed=1000 + ep)
        cost, util, curtail, soc_tr, cost_tr = run_episode(env_dqn, dqn_policy)
        dqn_costs.append(cost)
        dqn_utils.append(util)
        dqn_curtails.append(curtail)
        if ep == 0:
            dqn_soc_trace, dqn_cost_trace = soc_tr, cost_tr

        env_rule = MicrogridEnv(seed=1000 + ep)
        cost, util, curtail, soc_tr, cost_tr = run_episode(env_rule, rule_based_action)
        rule_costs.append(cost)
        rule_utils.append(util)
        rule_curtails.append(curtail)
        if ep == 0:
            rule_soc_trace, rule_cost_trace = soc_tr, cost_tr

    dqn_avg_cost = np.mean(dqn_costs)
    dqn_avg_util = np.mean(dqn_utils) * 100
    dqn_avg_curtail_wh = np.mean(dqn_curtails)
    rule_avg_cost = np.mean(rule_costs)
    rule_avg_util = np.mean(rule_utils) * 100
    rule_avg_curtail_wh = np.mean(rule_curtails)

    cost_improvement = (rule_avg_cost - dqn_avg_cost) / rule_avg_cost * 100 if rule_avg_cost != 0 else 0
    util_improvement_pp = dqn_avg_util - rule_avg_util

    print("=" * 55)
    print(f"Averaged over {N_EVAL_EPISODES} evaluation episodes (fixed seeds, both policies see identical days)")
    print("=" * 55)
    print(f"{'Metric':<32}{'Rule-based':>11}{'DQN':>11}")
    print(f"{'Avg daily grid cost ($)':<32}{rule_avg_cost:>11.4f}{dqn_avg_cost:>11.4f}")
    print(f"{'True solar utilization (%)':<32}{rule_avg_util:>11.2f}{dqn_avg_util:>11.2f}")
    print(f"{'Avg solar curtailed (Wh/day)':<32}{rule_avg_curtail_wh:>11.2f}{dqn_avg_curtail_wh:>11.2f}")
    print("-" * 55)
    print(f"Cost improvement (DQN vs rule-based): {cost_improvement:+.2f}%")
    print(f"Solar utilization change: {util_improvement_pp:+.2f} percentage points")

    # plot
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    hours = range(len(dqn_soc_trace))
    axes[0].plot(hours, dqn_soc_trace, label="DQN", color="tab:blue")
    axes[0].plot(range(len(rule_soc_trace)), rule_soc_trace, label="Rule-based", color="tab:orange")
    axes[0].set_ylabel("Battery SOC")
    axes[0].set_title("Battery SOC over a sample day")
    axes[0].legend()

    hours2 = range(len(dqn_cost_trace))
    axes[1].plot(hours2, np.cumsum(dqn_cost_trace), label="DQN", color="tab:blue")
    axes[1].plot(range(len(rule_cost_trace)), np.cumsum(rule_cost_trace), label="Rule-based", color="tab:orange")
    axes[1].set_ylabel("Cumulative grid cost ($)")
    axes[1].set_xlabel("Hour of day")
    axes[1].set_title("Cumulative Cost over a sample day")
    axes[1].legend()

    plt.tight_layout()
    import os
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/comparison_plot.png", dpi=120)
    print("\nSaved comparison plot to results/comparison_plot.png")


if __name__ == "__main__":
    main()
