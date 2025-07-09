import pandas as pd, numpy as np, os
from pathlib import Path

downloads = str(Path.home() / "Downloads")
if not os.path.isdir(downloads):
    downloads = os.getcwd()

csv_path = input("Enter CSV file path: ").strip()
df = pd.read_csv(csv_path)

if 'New_VehicleID' in df.columns:
    merged_file = True
    id_col = 'New_VehicleID'
    print("✓ Detected merged file (using New_VehicleID)")
else:
    merged_file = False
    id_col = 'VehicleID'
    if id_col not in df.columns:
        raise ValueError("CSV must contain a 'VehicleID' column.")
    print("✓ Regular file (using VehicleID)")

if 'VehicleID' in df.columns:
    df['VehicleType'] = (
        df['VehicleID'].astype(str)
          .str.extract(r'(^[A-Za-z]+)')
          .fillna('Unknown')[0]
    )
else:
    df['VehicleType'] = (
        df[id_col].astype(str)
          .str.extract(r'(^[A-Za-z]+)')
          .fillna('Unknown')[0]
    )

num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
            if not df[c].isna().all()]

if not num_cols:
    raise ValueError("No numeric columns detected.")

def cv(x):
    m = x.mean()
    return np.nan if m == 0 else x.std(ddof=0) / m

agg = {c: ['mean', 'min', 'max', 'std', ('cv', cv)] for c in num_cols}

type_stats = (
    df.groupby('VehicleType', sort=False)
      .agg(agg)
      .round(4)
)
type_stats.columns = [f'{col}_{stat}' for col, stat in type_stats.columns]
type_stats_path = os.path.join(downloads, 'VehicleType_Statistics.csv')
type_stats.to_csv(type_stats_path)

vid_stats = (
    df.groupby(id_col, sort=False)
      .agg(agg)
      .round(4)
)
vid_stats.columns = [f'{col}_{stat}' for col, stat in vid_stats.columns]
vid_stats.reset_index(inplace=True)
vid_stats.insert(0, id_col, vid_stats.pop(id_col))
vid_stats_path = os.path.join(downloads, 'VehicleID_Statistics.csv')
vid_stats.to_csv(vid_stats_path, index=False)

print("✓ Done! Two CSVs saved to Downloads folder.")
