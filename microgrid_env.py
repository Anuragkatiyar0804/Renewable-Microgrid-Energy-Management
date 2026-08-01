"""
Microgrid Environment for the DRL Energy Management project.

Simulates one day (24 one-hour steps) of a solar + battery + grid microgrid,
built to reflect real operating conditions, not just an abstract reward:

  1. High solar, load covered, battery has room  -> solar covers load AND
     charges battery; grid draw = 0.
  2. High solar, battery already full             -> excess solar is
     CURTAILED (wasted) - a real inefficiency the agent should try to avoid
     by keeping headroom ahead of peak solar hours.
  3. Low/no solar (night or cloudy weather), battery healthy -> battery
     discharges to cover load; grid draw = 0.
  4. Low/no solar, battery near CRITICAL SOC       -> discharging further is
     physically disallowed below the critical floor (protects battery
     lifespan); grid takes over the full load AND is the only thing that
     can recharge the battery back up.
  5. Battery recharged above the critical floor    -> system reverts to
     solar+battery-first operation; grid draw drops again.
  6. High load + weak solar (evening peak)         -> battery covers what it
     can, grid fills only the remaining shortfall (partial draw).
  7. Off-peak cheap grid price, room in battery     -> optional grid-to-
     battery arbitrage charging (buy cheap now, avoid buying expensive later).
  8. Variable/cloudy weather                       -> solar output for the
     whole day is scaled by a random "weather factor" sampled at reset(), so
     the agent must react to current conditions each hour, not memorize a
     fixed solar curve.

State (6 features): [pv_norm, load_norm, battery_soc, hour_sin, hour_cos, price_norm]
Actions (3): 0 = CHARGE/arbitrage, 1 = DISCHARGE, 2 = IDLE

Original implementation written for this project.
"""

import numpy as np


class MicrogridEnv:
    def __init__(self, episode_hours: int = 24, seed: int = None):
        self.episode_hours = episode_hours
        self.rng = np.random.default_rng(seed)

        # --- system parameters (sized so the battery is a REAL constraint,
        # not big enough to trivially cover the whole day) ---
        self.battery_capacity_kwh = 2.0
        self.battery_efficiency = 0.95
        self.solar_peak_w = 1000.0
        self.load_min_w = 100.0
        self.load_max_w = 800.0
        self.price_peak = 0.25       # $/kWh, 09:00-21:00
        self.price_offpeak = 0.10    # $/kWh, otherwise
        self.soc_min_safe = 0.20     # HARD critical floor - battery cannot discharge below this
        self.soc_max_safe = 0.95     # soft ceiling - discourage overcharging
        self.max_charge_rate_wh = 400.0  # per-hour cap on grid arbitrage charging

        # weather scenarios sampled once per episode (per simulated day):
        # clear (most common), partly cloudy, overcast
        self.weather_options = [1.0, 0.7, 0.35]
        self.weather_probs = [0.6, 0.25, 0.15]

        self.observation_space_shape = (6,)
        self.n_actions = 3

        self.hour = 0
        self.battery_soc = 0.5
        self.weather_factor = 1.0

    class _Box:
        def __init__(self, shape):
            self.shape = shape

    class _Discrete:
        def __init__(self, n):
            self.n = n

    @property
    def observation_space(self):
        return self._Box(self.observation_space_shape)

    @property
    def action_space(self):
        return self._Discrete(self.n_actions)

    # -----------------------------------------------------------------
    def _solar_output_w(self, hour: float) -> float:
        clear_sky_w = self.solar_peak_w * np.exp(-((hour - 12) ** 2) / (2 * 3.5 ** 2))
        return clear_sky_w * self.weather_factor

    def _load_w(self, hour: float) -> float:
        base = self.load_min_w
        morning = 300 * np.exp(-((hour - 8) ** 2) / (2 * 1.0 ** 2))
        lunch = 200 * np.exp(-((hour - 13) ** 2) / (2 * 1.0 ** 2))
        evening = 400 * np.exp(-((hour - 19.5) ** 2) / (2 * 1.5 ** 2))
        noise = self.rng.normal(0, 20)
        load = base + morning + lunch + evening + noise
        return float(np.clip(load, self.load_min_w, self.load_max_w))

    def _price(self, hour: float) -> float:
        return self.price_peak if 9 <= hour < 21 else self.price_offpeak

    def _get_state(self) -> np.ndarray:
        pv_w = self._solar_output_w(self.hour)
        load_w = self._load_w(self.hour)
        price = self._price(self.hour)

        pv_norm = pv_w / self.solar_peak_w
        load_norm = (load_w - self.load_min_w) / (self.load_max_w - self.load_min_w)
        price_norm = (price - self.price_offpeak) / (self.price_peak - self.price_offpeak)
        hour_sin = np.sin(2 * np.pi * self.hour / 24)
        hour_cos = np.cos(2 * np.pi * self.hour / 24)

        self._last_pv_w = pv_w
        self._last_load_w = load_w
        self._last_price = price

        return np.array(
            [pv_norm, load_norm, self.battery_soc, hour_sin, hour_cos, price_norm],
            dtype=np.float32,
        )

    # -----------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.hour = 0
        self.battery_soc = 0.5
        # sample today's weather once - stays fixed for the whole day,
        # like real weather does
        self.weather_factor = self.rng.choice(self.weather_options, p=self.weather_probs)
        return self._get_state()

    def step(self, action: int):
        pv_w = self._last_pv_w if hasattr(self, "_last_pv_w") else self._solar_output_w(self.hour)
        load_w = self._last_load_w if hasattr(self, "_last_load_w") else self._load_w(self.hour)
        price = self._last_price if hasattr(self, "_last_price") else self._price(self.hour)

        battery_capacity_wh = self.battery_capacity_kwh * 1000
        critical_floor_wh = self.soc_min_safe * battery_capacity_wh

        # --- PHYSICS (condition 1, 2): solar always serves load directly
        # first, excess always tries to charge the battery. No sane EMS
        # curtails free solar on purpose. ---
        solar_to_load = min(pv_w, load_w)
        excess_pv_w = max(0.0, pv_w - load_w)
        deficit_w = max(0.0, load_w - pv_w)

        room_wh = (1 - self.battery_soc) * battery_capacity_wh
        solar_to_battery_wh = min(excess_pv_w * self.battery_efficiency, room_wh)
        solar_curtailed_w = max(0.0, excess_pv_w - solar_to_battery_wh / self.battery_efficiency)
        room_remaining_wh = room_wh - solar_to_battery_wh

        # --- HARD CONSTRAINT (condition 4): the battery physically cannot
        # discharge below the critical floor, regardless of what the agent
        # chooses. This is what forces the grid to step in near-empty. ---
        energy_above_floor_wh = max(0.0, self.battery_soc * battery_capacity_wh - critical_floor_wh)

        grid_arbitrage_wh = 0.0
        battery_discharge_wh = 0.0
        grid_w = deficit_w  # default: grid covers whatever solar didn't (conditions 3, 5, 6)

        if action == 0:  # CHARGE / arbitrage (condition 7)
            if price <= self.price_offpeak + 1e-9 and room_remaining_wh > 0:
                grid_arbitrage_wh = min(self.max_charge_rate_wh, room_remaining_wh)
            # deficit met purely by grid; battery not discharged

        elif action == 1:  # DISCHARGE (conditions 3, 5, 6) - capped by the critical floor
            battery_discharge_wh = min(deficit_w, energy_above_floor_wh * self.battery_efficiency)
            grid_w = max(0.0, deficit_w - battery_discharge_wh)

        # action == 2 IDLE: deficit purely from grid, no discharge, no arbitrage

        grid_w += grid_arbitrage_wh
        battery_delta_wh = (
            solar_to_battery_wh + grid_arbitrage_wh
            - (battery_discharge_wh / self.battery_efficiency if battery_discharge_wh > 0 else 0.0)
        )

        self.battery_soc += battery_delta_wh / battery_capacity_wh
        self.battery_soc = float(np.clip(self.battery_soc, 0.0, 1.0))

        grid_cost = (grid_w / 1000.0) * price
        solar_bonus = (solar_to_load / 1000.0) * 0.05
        curtail_penalty = (solar_curtailed_w / 1000.0) * 0.05
        overcharge_pen = 0.5 if self.battery_soc > self.soc_max_safe else 0.0

        reward = -grid_cost + solar_bonus - curtail_penalty - overcharge_pen

        self.hour += 1
        done = self.hour >= self.episode_hours

        info = {
            "soc": self.battery_soc,
            "grid_cost": grid_cost,
            "solar_used_w": solar_to_load + solar_to_battery_wh,
            "solar_curtailed_w": solar_curtailed_w,
            "grid_w": grid_w,
            "pv_w": pv_w,
            "load_w": load_w,
            "price": price,
            "weather_factor": self.weather_factor,
            "hit_critical_floor": bool(energy_above_floor_wh <= 1e-6 and deficit_w > 0),
        }

        next_state = self._get_state() if not done else np.zeros(6, dtype=np.float32)
        return next_state, float(reward), done, info
