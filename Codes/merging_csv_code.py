import pandas as pd
from google.colab import files
import io

uploaded = files.upload()          
dfs = []
for idx, fname in enumerate(uploaded.keys(), start=1):       
    df = pd.read_csv(io.BytesIO(uploaded[fname]))

    if 'VehicleID' not in df.columns:
        raise ValueError(f"'VehicleID' column missing in {fname}")

    df['New_VehicleID'] = df['VehicleID'].astype(str) + f"_F{idx}"
    cols = df.columns.tolist()
    cols.insert(cols.index('VehicleID') + 1, cols.pop(cols.index('New_VehicleID')))
    df = df[cols]

    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)
out_name = "Merged_Vehicles.csv"
merged_df.to_csv(out_name, index=False)
files.download(out_name)
