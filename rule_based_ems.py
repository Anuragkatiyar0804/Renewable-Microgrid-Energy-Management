"""
Threshold-based rule EMS - baseline to compare the DQN agent against.

Simple, realistic logic a homeowner might set manually:
    - If battery is getting low (SOC < 35%) and grid price is currently
      off-peak/cheap -> CHARGE (top up affordably before it's needed)
    - Otherwise -> DISCHARGE (use stored + solar energy to avoid buying
      grid power; the environment's critical-SOC floor protects the
      battery automatically, so this is safe even when SOC is already low)
"""

import numpy as np
from microgrid_env import MicrogridEnv


def rule_based_action(state: np.ndarray) -> int:
    pv_norm, load_norm, soc, hour_sin, hour_cos, price_norm = state
    is_offpeak = price_norm < 0.5
    if soc < 0.35 and is_offpeak:
        return 0  # CHARGE / arbitrage
    return 1  # DISCHARGE


def run_rule_based_episode(env: MicrogridEnv):
    state = env.reset()
    total_cost = 0.0
    total_solar_used = 0.0
    total_pv = 0.0
    done = False
    while not done:
        action = rule_based_action(state)
        state, reward, done, info = env.step(action)
        total_cost += info["grid_cost"]
        total_solar_used += info["solar_used_w"]
        total_pv += info["pv_w"]
    utilization = total_solar_used / total_pv if total_pv > 0 else 0
    return total_cost, utilization


if __name__ == "__main__":
    env = MicrogridEnv(seed=42)
    cost, util = run_rule_based_episode(env)
    print(f"Rule-based baseline | Cost: ${cost:.3f} | Solar utilization: {util*100:.1f}%")
