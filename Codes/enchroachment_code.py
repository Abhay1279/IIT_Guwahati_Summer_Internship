import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import io

uploaded = files.upload()
vehicle_file = [f for f in uploaded if 'Merged' in f or 'smoothen' in f][0]
road_file = [f for f in uploaded if 'Homography' in f][0]
vehicle_df = pd.read_csv(io.BytesIO(uploaded[vehicle_file]))
road_df = pd.read_csv(io.BytesIO(uploaded[road_file]))

road_df['Center_N'] = (road_df['Left_N'] + road_df['Right_N']) / 2
road_df['Center_E'] = (road_df['Left_E'] + road_df['Right_E']) / 2
centerline_coords = road_df[['Center_N', 'Center_E']].to_numpy()

def compute_signed_deviation(vehicle_points, centerline):
    result = []
    for p in vehicle_points:
        min_dist = float('inf')
        sign = 1
        for i in range(len(centerline) - 1):
            a, b = centerline[i], centerline[i + 1]
            ap = p - a
            ab = b - a
            t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
            proj = a + t * ab
            dist = np.linalg.norm(p - proj)
            if dist < min_dist:
                min_dist = dist
                cross = ab[0]*ap[1] - ab[1]*ap[0]
                sign = 1 if cross < 0 else -1
        result.append(sign * min_dist)
    return result

def compute_encroachment_with_label(df, centerline_coords):
    df = df.sort_values('Time').reset_index(drop=True)
    vehicle_points = df[['Y_Smt', 'X_Smt']].to_numpy()
    df['Signed_Deviation'] = compute_signed_deviation(vehicle_points, centerline_coords)
    initial_sign = np.sign(df['Signed_Deviation'].iloc[:5].mean())
    sign_series = np.sign(df['Signed_Deviation'])
    enc_start_idx = None
    enc_end_idx = None
    lane_behavior = "In Lane"
    for i in range(1, len(df)):
        if sign_series[i] != initial_sign and sign_series[i-1] == initial_sign:
            enc_start_idx = i
            lane_behavior = "Encroachment Lane"
            break
    if enc_start_idx is not None:
        for j in range(enc_start_idx + 1, len(df)):
            if sign_series[j] == initial_sign:
                enc_end_idx = j
                break
        if enc_end_idx is None:
            enc_end_idx = len(df)
        enc_df = df.iloc[enc_start_idx:enc_end_idx].copy()
        enc_df['Abs_Deviation'] = np.abs(enc_df['Signed_Deviation'])
        coords = enc_df[['X_Smt', 'Y_Smt']].to_numpy()
        enc_len = np.sum(np.linalg.norm(coords[1:] - coords[:-1], axis=1))
        enc_area = np.trapz(enc_df['Abs_Deviation'], dx=0.04)
        enc_time = (enc_end_idx - enc_start_idx) * 0.04
    else:
        enc_len = enc_area = enc_time = 0
    return pd.Series({
        'VehicleID': df['VehicleID'].iloc[0],
        'Initial_Lane_Sign': initial_sign,
        'Encroachment_Area_m2': round(enc_area, 3),
        'Encroachment_Length_m': round(enc_len, 3),
        'Encroachment_Time_s': round(enc_time, 3),
        'Lane_Behavior': lane_behavior
    })

summary = vehicle_df.groupby('VehicleID').apply(
    lambda x: compute_encroachment_with_label(x.copy(), centerline_coords)
).reset_index(drop=True)

summary.to_csv("/content/Encroachment_Episode_With_Behavior.csv", index=False)
files.download("/content/Encroachment_Episode_With_Behavior.csv")

from IPython.display import display
display(summary)
