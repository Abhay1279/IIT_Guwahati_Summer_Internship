import pandas as pd, numpy as np, re, os
from pathlib import Path

downloads = str(Path.home() / "Downloads")
if not os.path.isdir(downloads):
    downloads = os.getcwd()

overtaking_path = input("Enter path to overtaking-summary CSV: ").strip()
vehicle_path = input("Enter path to trajectory/processed CSV: ").strip()

overtaking_df = pd.read_csv(overtaking_path)
vehicle_df = pd.read_csv(vehicle_path)

overtaking_df.columns = overtaking_df.columns.str.strip()
vehicle_df.columns = vehicle_df.columns.str.strip()

for cand in ['Opposing_Vehicles_During_Overtake', 'Opposing_Vehicle_IDs', 'Opposing_Vehicles', 'OppVehicleIDs']:
    if cand in overtaking_df.columns:
        opp_col = cand
        break
else:
    opp_col = 'Opposing_Vehicles_During_Overtake'
    overtaking_df[opp_col] = ''

for cand in ['Distance from centreline', 'Distance_from_Centreline', 'distance from centreline', 'distance_from_centreline']:
    if cand in vehicle_df.columns:
        dist_col = cand
        break
else:
    raise ValueError("No centreline-distance column found in trajectory data.")

overtaking_df[opp_col] = overtaking_df[opp_col].fillna('').astype(str)
overtaking_df['Category'] = np.where(
    overtaking_df[opp_col].str.strip() != '',
    'Overtaking with oncoming vehicles',
    'Overtaking without oncoming vehicles'
)

all_ids = vehicle_df['VehicleID'].unique()
overtaking_ids = overtaking_df['Overtaking_Vehicle_ID'].unique()
non_ids = np.setdiff1d(all_ids, overtaking_ids)

threshold = 1.5

def classify_non_overtaker(vid):
    mean_dev = vehicle_df.loc[vehicle_df['VehicleID'] == vid, dist_col].abs().mean()
    return ('No overtaking and in lane with oncoming vehicles'
            if mean_dev <= threshold else
            'Free flowing')

non_df = pd.DataFrame({'VehicleID': non_ids})
non_df['Category'] = non_df['VehicleID'].map(classify_non_overtaker)

ovt_res = overtaking_df[['Overtaking_Vehicle_ID', 'Category']].rename(
    columns={'Overtaking_Vehicle_ID': 'VehicleID'}
)
final_df = (
    pd.concat([ovt_res, non_df], ignore_index=True)
      .sort_values('VehicleID', kind='mergesort')
      .reset_index(drop=True)
)

out_name = os.path.join(downloads, 'Vehicle_Category_Summary.csv')
final_df.to_csv(out_name, index=False)
print("✓ Done! Output saved to:", out_name)
