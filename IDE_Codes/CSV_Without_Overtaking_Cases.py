import pandas as pd, numpy as np, re
from tqdm import tqdm

traj_path = input("Enter path to merged trajectory CSV: ").strip()
sum_path = input("Enter path to overtaking summary CSV: ").strip()

traj = pd.read_csv(traj_path)
summ = pd.read_csv(sum_path)

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
out = 'Final_Merged_Trajectory_CLEAN.csv'
clean.to_csv(out, index=False)

print(f'Removed {len(idx_to_drop):,} rows ({len(idx_to_drop)/len(traj):.2%} of total).')
print("✓ Done! Output saved as:", out)
