import pandas as pd, numpy as np, io
from google.colab import files

uploaded = files.upload()
veh_file = next(f for f in uploaded if 'final_merged_trajectory'.lower() in f.lower())
df       = pd.read_csv(io.BytesIO(uploaded[veh_file]))

signed_col = 'Distance from centreline'             
veh_groups = df.groupby('VehicleID')
veh_ids    = df['VehicleID'].unique()

abs_dev    = df[signed_col].abs().dropna()
half_lane  = np.percentile(abs_dev, 90)                
noise_band = abs_dev[abs_dev < np.percentile(abs_dev, 10)]
lane_thr_m = (np.percentile(noise_band, 90)             
              if len(noise_band) else 0.05)
proximity_m = 1.1 * (2 * half_lane)                     

print(f'λ half-lane≈{half_lane:.2f} m | '
      f'noise±{lane_thr_m:.2f} m | '
      f'proximity={proximity_m:.2f} m')

def seg_metrics(seg: pd.DataFrame):
    if len(seg) < 2: return 0., 0.
    d = np.hypot(np.diff(seg['X_Smt']), np.diff(seg['Y_Smt'])).sum()
    t = seg['Time'].iat[-1] - seg['Time'].iat[0]
    return round(d, 2), round(t, 2)

coord  = lambda r: (round(r['X_Smt'], 2), round(r['Y_Smt'], 2))
result = []

for fast_id in veh_ids:
    fast = veh_groups.get_group(fast_id).sort_values('Time').reset_index(drop=True)
    fast_dir  = fast['Direction'].iat[0]
    fast_type = fast_id.split('_')[0]

    for slow_id in veh_ids:
        if slow_id == fast_id: continue
        slow = veh_groups.get_group(slow_id).sort_values('Time').reset_index(drop=True)
        if slow['Direction'].iat[0] != fast_dir: continue  # same travel dir only

        t0, t1 = max(fast['Time'].iat[0], slow['Time'].iat[0]), \
                 min(fast['Time'].iat[-1], slow['Time'].iat[-1])
        if t1 <= t0: continue

        fast_win = fast[(fast['Time'] >= t0) & (fast['Time'] <= t1)].reset_index(drop=True)
        slow_win = slow[(slow['Time'] >= t0) & (slow['Time'] <= t1)].reset_index(drop=True)
        if len(fast_win) < 2 or len(slow_win) < 2: continue

        aligned = pd.merge_asof(
            fast_win[['Time','X_Smt','Y_Smt']],
            slow_win[['Time','X_Smt','Y_Smt']],
            on='Time', direction='nearest', suffixes=('_f','_s')
        ).dropna()

        if len(aligned) < 2: continue

        gaps = np.hypot(aligned['X_Smt_f'] - aligned['X_Smt_s'],
                        aligned['Y_Smt_f'] - aligned['Y_Smt_s']).values

        if not (gaps[0] - gaps.min() > lane_thr_m and gaps.min() < proximity_m):
            continue   

        dev  = fast_win[signed_col].values
        sign = np.sign(np.clip(dev, -lane_thr_m, lane_thr_m))
        cross_pts = np.where(np.diff(sign) != 0)[0]
        if cross_pts.size == 0: continue     

        ci = cross_pts[0]                   
        ret_cand = cross_pts[1:]
        returned = ret_cand.size > 0
        ri = ret_cand[-1] if returned else None
        before = fast_win.iloc[:ci+1]
        during = fast_win.iloc[ci+1:ri+1] if returned else fast_win.iloc[ci+1:]
        after  = fast_win.iloc[ri+1:] if returned else pd.DataFrame(columns=fast_win.columns)
        d1, t1 = seg_metrics(before)
        d2, t2 = seg_metrics(during)
        d3, t3 = seg_metrics(after)
        tot_d, tot_t = round(d1+d2+d3, 2), round(t1+t2+t3, 2)
        sc = coord(fast_win.iloc[ci])
        ec = coord(fast_win.iloc[ri]) if returned else coord(fast_win.iloc[-1])
        opp_ids, min_gap, min_dt = [], np.inf, np.inf
        t_ret = fast_win['Time'].iat[ri if returned else -1]
        for opp_id in veh_ids:
            if opp_id in (fast_id, slow_id): continue
            opp = veh_groups.get_group(opp_id)
            if opp['Direction'].iat[0] == fast_dir: continue
            if opp[(opp['Time'] >= fast_win['Time'].iat[0]) &
                   (opp['Time'] <= fast_win['Time'].iat[-1])].empty: continue
            opp_ids.append(opp_id)
            idx = (opp['Time'] - t_ret).abs().idxmin()
            gap = np.hypot(opp.at[idx,'X_Smt'] - fast_win.iloc[-1]['X_Smt'],
                           opp.at[idx,'Y_Smt'] - fast_win.iloc[-1]['Y_Smt'])
            dt  = abs(opp.at[idx,'Time'] - t_ret)
            if gap < min_gap: min_gap, min_dt = gap, dt

        result.append({
            "Overtaking_Vehicle_ID"       : fast_id,
            "Overtaking_Vehicle_Type"     : fast_type,
            "Overtaken_Vehicle_ID"        : slow_id,
            "Overtaken_Count"             : 1,
            "Overtaken_IDs"               : slow_id,
            "Overtake_Distance_m"         : tot_d,
            "Overtake_Time_s"             : tot_t,
            "Direction"                   : fast_dir,
            "Opposing_In_Window"          : ','.join(opp_ids),
            "To_CL_Before_Dist_m"         : d1,
            "To_CL_Before_Time_s"         : t1,
            "Pass+To_CL_Dist_m"           : round(d1+d2,2),
            "Pass+To_CL_Time_s"           : round(t1+t2,2),
            "To_LC_After_Dist_m"          : d3,
            "To_LC_After_Time_s"          : t3,
            "ReturnGap_Dist_m"            : round(min_gap,2) if np.isfinite(min_gap) else np.nan,
            "ReturnGap_Time_s"            : round(min_dt,2)  if np.isfinite(min_dt) else np.nan,
            "Start_XY"                    : sc,
            "End_XY"                      : ec
        })

summary = pd.DataFrame(result)
out = "Final_Overtaking_Summary_With_Overtaken_ID_and_Coords.csv"
summary.to_csv(out, index=False)
files.download(out)
print(f"✓ Completed – {len(summary)} overtakes detected")
