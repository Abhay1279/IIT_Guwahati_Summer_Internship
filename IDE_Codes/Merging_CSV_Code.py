import pandas as pd

file_paths = input("Enter paths to vehicle CSV files (comma-separated): ").strip().split(",")
file_paths = [f.strip() for f in file_paths]

dfs = []
for idx, fname in enumerate(file_paths, start=1):
    df = pd.read_csv(fname)
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
print("✓ Done! Output saved as:", out_name)
