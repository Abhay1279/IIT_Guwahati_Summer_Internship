import pandas as pd, numpy as np, io, time
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from google.colab import files

plt.rcParams["figure.dpi"] = 150

uploaded = files.upload()                   

centreline_file = [f for f in uploaded if "centreline" in f.lower()][0]
homography_file = [f for f in uploaded if "homography" in f.lower()][0]
trajectory_file = [f for f in uploaded if "merged"     in f.lower()][0]

centreline_df = pd.read_csv(io.BytesIO(uploaded[centreline_file]), header=None)
if centreline_df.shape[1] < 3:
    raise ValueError("Centreline file must have at least 3 columns (Label, X, Y).")
col_map = ["Label", "X", "Y"] + [f"Col{i}" for i in range(4, centreline_df.shape[1] + 1)]
centreline_df.columns = col_map[:centreline_df.shape[1]]
homography_df = pd.read_csv(io.BytesIO(uploaded[homography_file]))
veh_df = pd.read_csv(io.BytesIO(uploaded[trajectory_file]))

left_x  = pd.to_numeric(homography_df["Left_E"],  errors="coerce")
left_y  = pd.to_numeric(homography_df["Left_N"],  errors="coerce")
right_x = pd.to_numeric(homography_df["Right_E"], errors="coerce")
right_y = pd.to_numeric(homography_df["Right_N"], errors="coerce")

centre_x = (left_x + right_x) / 2
centre_y = (left_y + right_y) / 2

clean_df = pd.DataFrame({"X": centre_x, "Y": centre_y}).dropna().reset_index(drop=True)
left_x_clean,  left_y_clean  = left_x.dropna().reset_index(drop=True),  left_y.dropna().reset_index(drop=True)
right_x_clean, right_y_clean = right_x.dropna().reset_index(drop=True), right_y.dropna().reset_index(drop=True)

seg_len         = np.hypot(clean_df["X"].diff(), clean_df["Y"].diff()).fillna(0)
cumulative_dist = seg_len.cumsum()
total_length, centre_pos = cumulative_dist.iloc[-1], cumulative_dist.iloc[-1] / 2
circular_start, circular_end = np.clip(centre_pos - 25, 0, total_length), np.clip(centre_pos + 25, 0, total_length)
trans1_start,  trans2_end   = np.clip(circular_start - 50, 0, total_length), np.clip(circular_end + 50, 0, total_length)
idx_trans1_start = (cumulative_dist - trans1_start ).abs().idxmin()
idx_circ_start   = (cumulative_dist - circular_start).abs().idxmin()
idx_circ_end     = (cumulative_dist - circular_end  ).abs().idxmin()
idx_trans2_end   = (cumulative_dist - trans2_end    ).abs().idxmin()
idx_centre       = (cumulative_dist - centre_pos    ).abs().idxmin()
centre_coords = clean_df[["X", "Y"]].to_numpy()

def project_dist(point, polyline):
    """Return cumulative-distance (m) along *polyline* at perpendicular projection of *point*."""
    min_d, best_cum, cum = np.inf, 0.0, 0.0
    for p1, p2 in zip(polyline[:-1], polyline[1:]):
        v = p2 - p1
        if not v.any():                   # zero-length segment
            continue
        t = np.clip(np.dot(point - p1, v) / np.dot(v, v), 0, 1)
        proj = p1 + t * v
        d    = np.linalg.norm(point - proj)
        if d < min_d:
            min_d, best_cum = d, cum + np.linalg.norm(p1 - proj)
        cum += np.linalg.norm(v)
    return best_cum

veh_df["Cumulative_Dist"] = [
    project_dist(np.array([row["X_Smt"], row["Y_Smt"]]), centre_coords)
    for _, row in veh_df.iterrows()
]

def region_of(d):
    if trans1_start <= d < circular_start: return "Transition 1"
    if circular_start <= d < circular_end: return "Circular"
    if circular_end <= d <= trans2_end:    return "Transition 2"
    return "Tangent"

veh_df["Region"] = veh_df["Cumulative_Dist"].apply(region_of)

def smooth_xy(x, y, n=400):
    if len(x) < 4:
        return x.values, y.values
    s = np.hypot(np.diff(x), np.diff(y)).cumsum()
    s = np.insert(s, 0, 0)
    s_new = np.linspace(0, s[-1], n)
    return CubicSpline(s, x)(s_new), CubicSpline(s, y)(s_new)

plt.figure(figsize=(16, 8))

if idx_trans1_start > 0:
    x, y = smooth_xy(clean_df["X"].iloc[:idx_trans1_start+1],
                     clean_df["Y"].iloc[:idx_trans1_start+1])
    plt.plot(x, y, lw=3, color="royalblue", label="Tangent (Entry)")
    lx, ly = smooth_xy(left_x_clean.iloc[:idx_trans1_start+1],
                       left_y_clean.iloc[:idx_trans1_start+1])
    rx, ry = smooth_xy(right_x_clean.iloc[:idx_trans1_start+1],
                       right_y_clean.iloc[:idx_trans1_start+1])
    plt.plot(lx, ly, lw=2, color="royalblue");  plt.plot(rx, ry, lw=2, color="royalblue")

x, y = smooth_xy(clean_df["X"].iloc[idx_trans1_start:idx_circ_start+1],
                 clean_df["Y"].iloc[idx_trans1_start:idx_circ_start+1])
plt.plot(x, y, lw=3, color="orange", label="Transition 1")
lx, ly = smooth_xy(left_x_clean.iloc[idx_trans1_start:idx_circ_start+1],
                   left_y_clean.iloc[idx_trans1_start:idx_circ_start+1])
rx, ry = smooth_xy(right_x_clean.iloc[idx_trans1_start:idx_circ_start+1],
                   right_y_clean.iloc[idx_trans1_start:idx_circ_start+1])
plt.plot(lx, ly, lw=2, color="orange");  plt.plot(rx, ry, lw=2, color="orange")

x, y = smooth_xy(clean_df["X"].iloc[idx_circ_start:idx_circ_end+1],
                 clean_df["Y"].iloc[idx_circ_start:idx_circ_end+1])
plt.plot(x, y, lw=3, color="red", label="Circular")
lx, ly = smooth_xy(left_x_clean.iloc[idx_circ_start:idx_circ_end+1],
                   left_y_clean.iloc[idx_circ_start:idx_circ_end+1])
rx, ry = smooth_xy(right_x_clean.iloc[idx_circ_start:idx_circ_end+1],
                   right_y_clean.iloc[idx_circ_start:idx_circ_end+1])
plt.plot(lx, ly, lw=2, color="red");     plt.plot(rx, ry, lw=2, color="red")

x, y = smooth_xy(clean_df["X"].iloc[idx_circ_end:idx_trans2_end+1],
                 clean_df["Y"].iloc[idx_circ_end:idx_trans2_end+1])
plt.plot(x, y, lw=3, color="green", label="Transition 2")
lx, ly = smooth_xy(left_x_clean.iloc[idx_circ_end:idx_trans2_end+1],
                   left_y_clean.iloc[idx_circ_end:idx_trans2_end+1])
rx, ry = smooth_xy(right_x_clean.iloc[idx_circ_end:idx_trans2_end+1],
                   right_y_clean.iloc[idx_circ_end:idx_trans2_end+1])
plt.plot(lx, ly, lw=2, color="green");   plt.plot(rx, ry, lw=2, color="green")

if idx_trans2_end < len(clean_df) - 1:
    x, y = smooth_xy(clean_df["X"].iloc[idx_trans2_end:],
                     clean_df["Y"].iloc[idx_trans2_end:])
    plt.plot(x, y, lw=3, color="royalblue", label="Tangent (Exit)")
    lx, ly = smooth_xy(left_x_clean.iloc[idx_trans2_end:],
                       left_y_clean.iloc[idx_trans2_end:])
    rx, ry = smooth_xy(right_x_clean.iloc[idx_trans2_end:],
                       right_y_clean.iloc[idx_trans2_end:])
    plt.plot(lx, ly, lw=2, color="royalblue");  plt.plot(rx, ry, lw=2, color="royalblue")

x_bg, y_bg   = smooth_xy(clean_df["X"],      clean_df["Y"])
lx_bg, ly_bg = smooth_xy(left_x_clean,       left_y_clean)
rx_bg, ry_bg = smooth_xy(right_x_clean,      right_y_clean)
plt.plot(x_bg,  y_bg,  lw=1, color="grey",  zorder=1)
plt.plot(lx_bg, ly_bg, lw=1, color="grey",  zorder=1)
plt.plot(rx_bg, ry_bg, lw=1, color="grey",  zorder=1)
plt.scatter(clean_df["X"].iloc[idx_centre], clean_df["Y"].iloc[idx_centre],
            s=60, color="black", zorder=5, label="Centre point")

plt.title("Road Centreline & Regions", fontsize=16)
plt.xlabel("Easting (X)");  plt.ylabel("Northing (Y)")
plt.axis("equal");  plt.grid(ls="--", alpha=0.4);  plt.legend()

fig_name = "Centreline_Regions.png"
plt.savefig(fig_name, dpi=300, bbox_inches="tight")
plt.show()

output_csv = "VehicleTrajectory_Classified.csv"
veh_df.drop(columns="Cumulative_Dist").to_csv(output_csv, index=False)

files.download(output_csv)   
time.sleep(0.5)              
files.download(fig_name)     
