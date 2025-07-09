import pandas as pd
import numpy as np
import os
from pathlib import Path

downloads = str(Path.home() / "Downloads")
if not os.path.isdir(downloads):
    downloads = os.getcwd()

file_path = input("Enter path to the CSV file: ").strip()
df = pd.read_csv(file_path)

def compute_steering_metrics(angles, threshold=0.1):
    angles = angles.dropna().values
    if len(angles) < 3:
        return np.nan, np.nan, np.nan, np.nan
    ASA = np.mean(np.abs(angles))
    SDSA = np.std(angles)
    SAR = np.max(angles) - np.min(angles)
    reversals = 0
    for i in range(1, len(angles) - 1):
        if (angles[i] - angles[i - 1]) * (angles[i + 1] - angles[i]) < 0:
            if abs(angles[i + 1] - angles[i]) > threshold:
                reversals += 1
    return ASA, SDSA, SAR, reversals

results = []
for vehicle_id, group in df.groupby("VehicleID"):
    ASA, SDSA, SAR, SRR = compute_steering_metrics(group["Heading_Angle_Rad"])
    results.append({
        "VehicleID": vehicle_id,
        "ASA (rad)": ASA,
        "SDSA (rad)": SDSA,
        "SAR (rad)": SAR,
        "SRR (count)": SRR
    })

metrics_df = pd.DataFrame(results)
metrics_df = metrics_df.sort_values("VehicleID").reset_index(drop=True)
output_filename = os.path.join(downloads, "Steering_Metrics_Per_Vehicle.csv")
metrics_df.to_csv(output_filename, index=False)
print("✓ Done! Output saved to:", output_filename)
