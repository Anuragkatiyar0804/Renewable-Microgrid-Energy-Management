# AI-Driven Optimal Energy Management for Renewable Integrated Smart Microgrids

# What this project is about

I built an AI agent that manages a small microgrid - solar panel, battery, and grid
connection - and decides every hour what to do with the power: use solar directly,
charge the battery, discharge the battery, or pull from the grid. The goal is simple:
keep the lights on all the time, but spend as little money on grid electricity as
possible, and use as much free solar energy as I can.

I used Deep Reinforcement Learning (specifically DQN - Deep Q-Network) for this,
written from scratch in PyTorch. I did not use any pre-built RL library like
Stable-Baselines3, because I wanted to actually understand and be able to explain
every part of the algorithm - the neural network, the replay buffer, the target
network, all of it.

## How I thought about the problem

Before writing any code, I first listed out the real situations this system needs
to handle, the way it would actually happen in a real home or building:

1. **Sun is strong, load is low** - solar alone covers the load, and the extra
   power charges the battery. No grid power needed at all.
2. **Sun is strong but battery is already full** - the extra solar has nowhere to
   go and gets wasted (this is called curtailment). A smart system should try to
   avoid this by keeping some space in the battery ready for peak sun hours.
3. **No sun (night time, or a cloudy day) but battery has charge** - battery
   takes over and powers the load. Still no grid needed.
4. **No sun, and battery charge is almost empty** - here the battery is not
   allowed to go any lower (I set a hard limit at 20% charge, so it doesn't get
   damaged from over-discharging). At this point the grid has to step in, both to
   run the load and to recharge the battery.
5. **Battery is charged back up** - once it's above the critical point again, the
   system goes back to using solar and battery first, and the grid switches off.
6. **High load in the evening while sun is fading** - battery gives what it can,
   and the grid only fills in the remaining gap, not the whole thing.
7. **Off-peak hours with cheap electricity** - even without much sun, if the
   grid price is low, the system can choose to top up the battery a bit now, so
   it has more stored energy ready for the expensive hours later.
8. **Weather changes** - some days are clear, some are partly cloudy, some are
   overcast. I made the solar output change with weather each day, so the agent
   has to actually react to what's happening, not just memorize one fixed pattern.

I made sure my simulated environment actually enforces these conditions as real
physics (like the battery genuinely cannot go below 20% no matter what action is
picked), not just something I mention in the report without it actually being true
in the code.

## What's in this repo

- `microgrid_env.py` - my simulation of the microgrid (solar, load, battery,
  grid pricing, weather) - written by me for this project.
- `dqn_agent.py` - my DQN implementation - the neural network, replay buffer,
  and training logic, in PyTorch.
- `rule_based_ems.py` - a simple rule-based system (charge when cheap and low,
  otherwise discharge) that I use as a baseline to compare my AI agent against.
- `tune_hyperparams.py` - a script I used to search for good hyperparameters
  (learning rate, batch size, etc.) instead of guessing them.
- `train_agent.py` - trains the DQN agent.
- `evaluate.py` - runs both the DQN agent and the rule-based baseline on the
  same simulated days and compares their cost and solar usage.
- `app.py` - a Streamlit dashboard where I can run a simulated day and see the
  battery charge, cost, and actions taken, visually.
- `models/` - my trained model and training curve.
- `results/` - comparison plots between my agent and the baseline.

## My results

I compared my trained DQN agent against the simple rule-based system over 20
simulated days (same days for both, so it's a fair comparison):

| Metric | Rule-based | My DQN agent |
|---|---|---|
| Average daily grid cost | $0.103 | $0.010 |
| Solar utilization | 53.3% | 58.5% |
| Solar wasted (curtailed) per day | 3952 Wh | 3493 Wh |

My agent cut the average daily electricity cost by about 90% compared to the
simple rule-based approach, and it also wastes less free solar energy because
it manages the battery's available space better across the day.

I picked my hyperparameters using a small random search rather than just
guessing:
```
learning rate: 0.002
batch size: 32
target network update frequency: 1000 steps
epsilon decay: 5000 steps
discount factor (gamma): 0.95
```

## How to run it myself

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt

python train_agent.py --steps 60000     # train the agent
python evaluate.py                      # compare against the rule-based baseline
streamlit run app.py                    # open the interactive dashboard
```

## Tools I used

Python, PyTorch, NumPy, Pandas, Matplotlib, and Streamlit for the dashboard.
