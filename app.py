
import streamlit as st
import numpy as np
import pandas as pd

from microgrid_env import MicrogridEnv
from dqn_agent import DQNAgent
from rule_based_ems import rule_based_action

st.set_page_config(page_title="Microgrid DRL Energy Management", layout="wide")

st.title("🔋 AI-Driven Renewable Microgrid Energy Management")
st.caption(
    "Deep Q-Network agent vs. a threshold-based rule EMS, simulated over a 24-hour day "
    "with solar generation, battery storage, and time-of-use grid pricing."
)
st.warning("This dashboard runs on the simulator. On real data the DQN performs about the same as a simple solar-first rule (up to about 1% better with a forecast). See the README for the real-data results.")
ACTION_NAMES = {0: "Charge", 1: "Discharge", 2: "Idle"}

with st.sidebar:
    st.header("Simulation settings")
    seed = st.number_input("Random seed (changes the simulated day)", min_value=0, max_value=9999, value=42)
    run_button = st.button("Run simulation", type="primary")


@st.cache_resource
def load_agent(model_path="models/dqn_microgrid.pt"):
    agent = DQNAgent(state_dim=6, action_dim=3)
    agent.load(model_path)
    return agent


def simulate(policy_fn, seed):
    env = MicrogridEnv(seed=seed)
    state = env.reset()
    rows = []
    done = False
    hour = 0
    while not done:
        action = policy_fn(state)
        state, reward, done, info = env.step(action)
        rows.append({
            "hour": hour,
            "action": ACTION_NAMES[action],
            "soc": info["soc"],
            "grid_cost": info["grid_cost"],
            "pv_w": info["pv_w"],
            "load_w": info["load_w"],
            "price": info["price"],
        })
        hour += 1
    df = pd.DataFrame(rows)
    df["cumulative_cost"] = df["grid_cost"].cumsum()
    return df


if run_button:
    agent = load_agent()
    dqn_policy = lambda s: agent.select_action(s, greedy=True)

    dqn_df = simulate(dqn_policy, seed)
    rule_df = simulate(rule_based_action, seed)

    dqn_total_cost = dqn_df["grid_cost"].sum()
    rule_total_cost = rule_df["grid_cost"].sum()
    improvement = (rule_total_cost - dqn_total_cost) / rule_total_cost * 100 if rule_total_cost else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("DQN daily cost", f"${dqn_total_cost:.3f}")
    col2.metric("Rule-based daily cost", f"${rule_total_cost:.3f}")
    col3.metric("Cost improvement (simulator only)", f"{improvement:+.1f}%")

    st.subheader("Battery State of Charge")
    soc_compare = pd.DataFrame({
        "hour": dqn_df["hour"],
        "DQN": dqn_df["soc"],
        "Rule-based": rule_df["soc"],
    }).set_index("hour")
    st.line_chart(soc_compare)

    st.subheader("Cumulative Grid Cost")
    cost_compare = pd.DataFrame({
        "hour": dqn_df["hour"],
        "DQN": dqn_df["cumulative_cost"],
        "Rule-based": rule_df["cumulative_cost"],
    }).set_index("hour")
    st.line_chart(cost_compare)

    st.subheader("Solar Generation vs Load (simulated day)")
    env_profile = pd.DataFrame({
        "hour": dqn_df["hour"],
        "Solar (W)": dqn_df["pv_w"],
        "Load (W)": dqn_df["load_w"],
    }).set_index("hour")
    st.line_chart(env_profile)

    st.subheader("DQN agent's hourly actions")
    st.dataframe(dqn_df[["hour", "action", "soc", "grid_cost", "price"]], use_container_width=True)
else:
    st.info("Set a seed and click **Run simulation** in the sidebar to compare the DQN agent against the rule-based EMS.")
