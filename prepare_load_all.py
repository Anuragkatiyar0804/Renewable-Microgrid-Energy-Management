import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

data = fetch_ucirepo(id=235).data.features
data.columns = [c.lower() for c in data.columns]

data["power_kw"] = pd.to_numeric(data["global_active_power"], errors="coerce")
data["dt"] = pd.to_datetime(
    data["date"].astype(str) + " " + data["time"].astype(str),
    dayfirst=True, errors="coerce")

hourly = (data.set_index("dt")["power_kw"].resample("60min").mean() * 1000).to_frame("load_w")
hourly["date"] = hourly.index.normalize()
days = hourly.groupby("date").filter(lambda g: len(g) == 24 and g["load_w"].notna().all())

arr = days["load_w"].values.reshape(-1, 24)
arr = arr * (215.0 / arr.mean())                 # same average size as the simulated home
dates = days.index[::24]                          # first hour of each day

out = np.column_stack([dates.year, dates.month, dates.day, arr])
np.savetxt("household_load_all.csv", out, delimiter=",", fmt="%.2f")

print("Saved", arr.shape[0], "real days to household_load_all.csv")
print(pd.Series(dates.year).value_counts().sort_index())