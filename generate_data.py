import pandas as pd
import numpy as np
import random

# reproducibility
random.seed(42)
np.random.seed(42)

n_rows = 100

# Base lat/lon region (roughly 53–55 N, 0–7 E)
lat_base, lon_base = 54.0, 3.5
latitudes = np.random.normal(lat_base, 0.8, n_rows)
longitudes = np.random.normal(lon_base, 2.0, n_rows)
latitudes = np.clip(latitudes, 52.5, 56.0)
longitudes = np.clip(longitudes, -0.5, 8.0)

# Weight (tons)
weights = np.round(np.random.uniform(0.25, 5.5, n_rows), 2)

# Collection time correlated with weight
collection_time = np.round(0.4 + weights * np.random.uniform(0.15, 0.35, n_rows), 2)
collection_time = np.clip(collection_time, 0.5, 1.8)

# Time windows (h)
# Random lower bounds in [8, 10], upper bounds +4 to +12h after lower bound
start_times = np.round(np.random.uniform(8.0, 10.0, n_rows), 1)
end_times = np.round(start_times + np.random.uniform(3.5, 12.0, n_rows), 1)
# Clip end times to [11, 21.5]
end_times = np.clip(end_times, 11.0, 21.5)
time_windows = [f"[{s},{e}]" for s, e in zip(start_times, end_times)]

# Construct DataFrame
df = pd.DataFrame({
    "No": range(1, n_rows+1),
    "Latitude (N)": np.round(latitudes, 6),
    "Longitude (E)": np.round(longitudes, 6),
    "Time Windows": time_windows,
    "Weight (tons)": weights,
    "Collection time (h)": collection_time
})

print(df.head(10))
