import pandas as pd, numpy as np, io
from google.colab import files
from tqdm import tqdm

uploaded  = files.upload()                         
csv_name  = next(iter(uploaded))
df        = pd.read_csv(io.BytesIO(uploaded[csv_name]))

if 'New_VehicleID' in df.columns:
    merged_file   = True
    id_col        = 'New_VehicleID'
    print("✓ Detected merged file (using New_VehicleID)")
else:
    merged_file   = False
    id_col        = 'VehicleID'
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
    return np.nan if m == 0 else x.std(ddof=0) / m   # population SD

agg = {c: ['mean', 'min', 'max', 'std', ('cv', cv)] for c in num_cols}

type_stats = (
    df.groupby('VehicleType', sort=False)
      .agg(agg)
      .round(4)
)
type_stats.columns = [f'{col}_{stat}' for col, stat in type_stats.columns]
type_stats.to_csv('VehicleType_Statistics.csv')
files.download('VehicleType_Statistics.csv')

vid_stats = (
    df.groupby(id_col, sort=False)
      .agg(agg)
      .round(4)
)
vid_stats.columns = [f'{col}_{stat}' for col, stat in vid_stats.columns]

vid_stats.reset_index(inplace=True)
vid_stats.insert(0, id_col, vid_stats.pop(id_col))   
vid_stats.to_csv('VehicleID_Statistics.csv', index=False)
files.download('VehicleID_Statistics.csv')

print("✓ Done!  Two CSVs downloaded.")
