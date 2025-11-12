import pandas as pd
import numpy as np
import random
import os

def generate_marine_debris_dataset(n_rows: int, seed: int = 42):
    """Generate a synthetic marine debris dataset with n_rows entries."""
    random.seed(seed)
    np.random.seed(seed)

    # Geographic region: North Sea area (approx 53–55° N, 0–7° E)
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

    # Time windows [start, end] in hours
    start_times = np.round(np.random.uniform(8.0, 9.0, n_rows), 1)
    end_times = np.round(start_times + np.random.uniform(3.5, 12.0, n_rows), 1)
    end_times = np.clip(end_times, 11.0, 21.5)
    time_windows = [f"[{s},{e}]" for s, e in zip(start_times, end_times)]

    # Construct DataFrame
    df = pd.DataFrame({
        "no.": range(1, n_rows + 1),
        "latitude(N)": np.round(latitudes, 6),
        "longitude(E)": np.round(longitudes, 6),
        "time_windows": time_windows,
        "weight(tons)": weights,
        "collection_time(h)": collection_time
    })

    return df

def save_multiple_datasets(sizes=(20, 50, 100), output_dir="datasets"):
    """Generate and save multiple marine debris datasets with given sizes."""
    os.makedirs(output_dir, exist_ok=True)

    for n in sizes:
        df = generate_marine_debris_dataset(n_rows=n, seed=42 + n)
        file_path = os.path.join(output_dir, f"marine_debris_{n}.csv")
        df.to_csv(file_path, index=False)
        print(f"✅ Saved {n}-row dataset to {file_path}")

    print("All datasets generated successfully!")

if __name__ == "__main__":
    save_multiple_datasets()
