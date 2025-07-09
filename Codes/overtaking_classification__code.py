import pandas as pd, numpy as np, io, re
from google.colab import files

print("📂  Upload the overtaking-summary CSV and the trajectory CSV:")
uploads = files.upload()

ovt_path, veh_path = None, None
for name in uploads:
    lname = name.lower()
    if re.search(r'overtaking.*summary', lname):
        ovt_path = name
    if ('processed_vehicles' in lname) or ('final_merged_trajectory' in lname):
        veh_path = name

if ovt_path is None:
    raise ValueError("Couldn’t locate an overtaking-summary CSV in your upload.")
if veh_path is None:
    raise ValueError("Couldn’t locate a vehicle-trajectory CSV in your upload.")

overtaking_df = pd.read_csv(io.BytesIO(uploads[ovt_path]))
vehicle_df    = pd.read_csv(io.BytesIO(uploads[veh_path]))
overtaking_df.columns = overtaking_df.columns.str.strip()
vehicle_df.columns    = vehicle_df.columns.str.strip()

for cand in ['Opposing_Vehicles_During_Overtake',
             'Opposing_Vehicle_IDs',
             'Opposing_Vehicles',
             'OppVehicleIDs']:
    if cand in overtaking_df.columns:
        opp_col = cand
        break
else:
    opp_col = 'Opposing_Vehicles_During_Overtake'
    overtaking_df[opp_col] = ''          

for cand in ['Distance from centreline', 'Distance_from_Centreline',
             'distance from centreline', 'distance_from_centreline']:
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

all_ids        = vehicle_df['VehicleID'].unique()
overtaking_ids = overtaking_df['Overtaking_Vehicle_ID'].unique()
non_ids        = np.setdiff1d(all_ids, overtaking_ids)

threshold = 1.5  

def classify_non_overtaker(vid):
    mean_dev = vehicle_df.loc[vehicle_df['VehicleID'] == vid, dist_col].abs().mean()
    return ('No overtaking and in lane with oncoming vehicles'
            if mean_dev <= threshold else
            'Free flowing')

non_df = pd.DataFrame({'VehicleID': non_ids})
non_df['Category'] = non_df['VehicleID'].map(classify_non_overtaker)
ovt_res = overtaking_df[['Overtaking_Vehicle_ID', 'Category']].rename(
            columns={'Overtaking_Vehicle_ID': 'VehicleID'})
final_df = (
    pd.concat([ovt_res, non_df], ignore_index=True)
      .sort_values('VehicleID', kind='mergesort')
      .reset_index(drop=True)
)

out_name = 'Vehicle_Category_Summary.csv'
final_df.to_csv(out_name, index=False)
files.download(out_name)

try:
    styled = (
        final_df.style
        .set_table_styles([{'selector': 'th',
                            'props': [('font-weight', 'bold'),
                                      ('background-color', '#d4d4d4')]}])
        .apply(lambda col: ['background-color: #f7f7f7' if i % 2 else ''
                            for i in range(len(col))], axis=0)
    )
    styled
except:
    pass
