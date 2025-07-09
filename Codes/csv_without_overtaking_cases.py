import pandas as pd, numpy as np, io, re
from tqdm import tqdm
from google.colab import files          

up         = files.upload()
traj_file  = next(f for f in up if re.search(r'trajectory|merged', f, re.I))
sum_file   = next(f for f in up if re.search(r'summary', f, re.I))
traj = pd.read_csv(io.BytesIO(up[traj_file]))
summ = pd.read_csv(io.BytesIO(up[sum_file]))

def xy_pair(txt):
    nums = re.findall(r'[-+]?\d*\.\d+|\d+', str(txt))
    return (float(nums[0]), float(nums[1])) if len(nums) >= 2 else None
OT_VIDCOL        = 'Overtaking_Vehicle_ID'
STARTCOL, ENDCOL = 'Start_XY', 'End_XY'
XCOL, YCOL, VIDCOL = 'X', 'Y', 'VehicleID'
idx_to_drop = set()
veh_groups  = {vid: g for vid, g in traj.groupby(VIDCOL)}
for _, ev in tqdm(summ.iterrows(), total=len(summ), desc='Marking segments'):
    vid = ev[OT_VIDCOL]
    if vid not in veh_groups:
        continue
    seg = veh_groups[vid]
    p1  = xy_pair(ev[STARTCOL])
    p2  = xy_pair(ev[ENDCOL])
    if (p1 is None) or (p2 is None):
        continue
    d1 = np.hypot(seg[XCOL] - p1[0], seg[YCOL] - p1[1])
    d2 = np.hypot(seg[XCOL] - p2[0], seg[YCOL] - p2[1])
    i1, i2 = d1.idxmin(), d2.idxmin()
    if i1 > i2:
        i1, i2 = i2, i1
    idx_to_drop.update(seg.loc[i1:i2].index)
clean = traj.drop(index=idx_to_drop, errors='ignore').reset_index(drop=True)
out   = 'Final_Merged_Trajectory_CLEAN.csv'
clean.to_csv(out, index=False)
print(f'Removed {len(idx_to_drop):,} rows '
      f'({len(idx_to_drop)/len(traj):.2%} of total).')
files.download(out)
