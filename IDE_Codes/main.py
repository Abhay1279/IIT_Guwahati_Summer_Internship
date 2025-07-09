import pandas as pd, numpy as np, matplotlib.pyplot as plt, math, re
from tqdm import tqdm
import os, shutil, warnings, sys
from pathlib import Path

plt.rcParams["figure.dpi"] = 200
warnings.filterwarnings("ignore", category=RuntimeWarning)

downloads = str(Path.home() / "Downloads")
if not os.path.isdir(downloads):
    downloads = os.getcwd()

vehicle_file    = input("Enter Vehicle Trajectory CSV file path: ").strip()
road_file       = input("Enter Homography (Road) CSV file path: ").strip()
centreline_file = input("Enter Centreline CSV file path: ").strip()

vehicle_df = pd.read_csv(vehicle_file)
road_df    = pd.read_csv(road_file)
centre_df  = pd.read_csv(centreline_file, header=None)

num_cols = ["Left_N","Left_E","Right_N","Right_E"]
road_df[num_cols] = road_df[num_cols].apply(pd.to_numeric, errors="coerce")

def acute_angle(deg):
    deg = (deg + 360) % 360
    if deg > 180: deg = 360 - deg
    if deg > 90:  deg = 180 - deg
    return deg

def proj_distance(p, a, b):
    ap, ab = p - a, b - a
    t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj), proj, ab

def min_seg_distances(points, segments):
    res = []
    for p in points:
        best = np.inf
        for a, b in zip(segments[:-1], segments[1:]):
            ab, ap = b - a, p - a
            denom  = np.hypot(*ab)
            perp   = abs(ab[0]*ap[1] - ab[1]*ap[0]) / denom if denom else np.linalg.norm(ap)
            best   = min(best, perp)
        res.append(best)
    return np.array(res)

def signed_deviation(points, centerline):
    res = []
    for p in points:
        best = np.inf; sgn = 1
        for a, b in zip(centerline[:-1], centerline[1:]):
            d, proj, ab = proj_distance(p, a, b)
            if d < best:
                best = d
                sgn  = 1 if (ab[0]*(p-a)[1] - ab[1]*(p-a)[0]) < 0 else -1
        res.append(sgn * best)
    return np.array(res)

def radius_three_pts(x, y):
    n=len(x); R=np.full(n, np.nan)
    for i in range(1, n-1):
        x1,y1,x2,y2,x3,y3 = x[i-1],y[i-1],x[i],y[i],x[i+1],y[i+1]
        a,b,c = math.hypot(x2-x1,y2-y1), math.hypot(x3-x2,y3-y2), math.hypot(x3-x1,y3-y1)
        s=(a+b+c)/2; A=max(s*(s-a)*(s-b)*(s-c),0)**0.5
        if A: R[i] = a*b*c/(4*A)
    return R

def densify_polyline(xy, step=0.5):
    xy = xy[~np.isnan(xy).any(axis=1)]
    if len(xy) < 2: return xy.copy()
    seg   = np.hypot(np.diff(xy[:,0]), np.diff(xy[:,1]))
    chain = np.insert(np.cumsum(seg), 0, 0.0)
    total = chain[-1]
    if total == 0 or np.isnan(total):
        return xy.copy()
    s_new = np.arange(0, total, step)
    N_new = np.interp(s_new, chain, xy[:,0])
    E_new = np.interp(s_new, chain, xy[:,1])
    return np.column_stack([N_new, E_new])

def direction_of_travel(y):
    return "Northbound" if y.tail(10).median() > y.head(10).median() else "Southbound"

def vtype(vid):
    tok = re.split(r"[^a-zA-Z0-9]", vid.lower())
    if "truck" in tok: return "Truck"
    if "lcv"   in tok: return "LCV"
    if "bike"  in tok or "motorbike" in tok: return "Bike"
    if "car"   in tok: return "Car"
    if "auto"  in tok: return "Auto"
    return "Other"

TYPE_COLOUR = {"Car":"tab:blue","Bike":"tab:orange","LCV":"tab:purple",
               "Truck":"tab:brown","Auto":"tab:green","Other":"tab:gray"}
WIDTH  = {"Truck":2.6,"LCV":2.1,"Bike":0.76,"Auto":1.4,"Car":2.5}
LENGTH = {"Truck":6.0,"LCV":3.5,"Bike":1.9,"Auto":2.5,"Car":2.5}
lookup = lambda d,vid:next((v for k,v in d.items() if k.lower() in vid.lower()), list(d.values())[-1])

road_df["Center_N"] = (road_df["Left_N"] + road_df["Right_N"]) / 2
road_df["Center_E"] = (road_df["Left_E"] + road_df["Right_E"]) / 2
road_df["LeftLaneCenter_N"]  = (road_df["Left_N"]  + road_df["Center_N"]) / 2
road_df["LeftLaneCenter_E"]  = (road_df["Left_E"]  + road_df["Center_E"]) / 2
road_df["RightLaneCenter_N"] = (road_df["Center_N"] + road_df["Right_N"]) / 2
road_df["RightLaneCenter_E"] = (road_df["Center_E"] + road_df["Right_E"]) / 2

centerline_pts    = densify_polyline(road_df[["Center_N","Center_E"]].to_numpy(), 0.5)
left_edge_pts     = densify_polyline(road_df[["Left_N","Left_E"]].to_numpy(),   0.5)
right_edge_pts    = densify_polyline(road_df[["Right_N","Right_E"]].to_numpy(),  0.5)
left_lane_center  = densify_polyline(road_df[["LeftLaneCenter_N","LeftLaneCenter_E"]].to_numpy(),  0.5)
right_lane_center = densify_polyline(road_df[["RightLaneCenter_N","RightLaneCenter_E"]].to_numpy(),0.5)

vehtraj_dir = os.path.join(downloads, "Vehicle_Trajectories")
yawplot_dir = os.path.join(downloads, "YawRate_Plots")
typeproj_dir = os.path.join(downloads, "VehicleType_Projections")
os.makedirs(vehtraj_dir, exist_ok=True)
os.makedirs(yawplot_dir, exist_ok=True)
os.makedirs(typeproj_dir, exist_ok=True)

rows_out, evasive_rows, processed_ids = [], [], []

for vid in tqdm(vehicle_df["VehicleID"].unique(), desc="Processing vehicles"):
    full = vehicle_df[vehicle_df["VehicleID"] == vid].reset_index(drop=True)
    if len(full) < 7: continue
    dev_full = signed_deviation(full[["Y_Smt","X_Smt"]].to_numpy(), centerline_pts)
    keep     = np.abs(dev_full) <= 3.5
    data, dev_c = full.loc[keep].reset_index(drop=True), dev_full[keep]
    if len(data) < 7: continue
    data, dev_c = data.iloc[3:-3].reset_index(drop=True), dev_c[3:-3]
    if data.empty: continue
    pts = data[["Y_Smt","X_Smt"]].to_numpy()
    dev_l   = min_seg_distances(pts, left_edge_pts)
    dev_r   = min_seg_distances(pts, right_edge_pts)
    dev_llc = min_seg_distances(pts, left_lane_center)
    dev_rlc = min_seg_distances(pts, right_lane_center)
    mask = (dev_l + dev_r) <= 7.5
    data, dev_c    = data.loc[mask].reset_index(drop=True), dev_c[mask]
    dev_l,dev_r    = dev_l[mask], dev_r[mask]
    dev_llc,dev_rlc= dev_llc[mask], dev_rlc[mask]
    if len(data) < 7: continue
    pts = data[["Y_Smt","X_Smt"]].to_numpy()
    vx, vy       = data["X_Smt"].diff().fillna(0), data["Y_Smt"].diff().fillna(0)
    heading_raw  = np.degrees(np.arctan2(vx, vy))
    theta_abs    = np.vectorize(acute_angle)(heading_raw)
    theta_rad    = np.radians(theta_abs)
    dt           = data["Time"].diff().fillna(0).values
    yaw_rate_rad = np.insert(np.diff(theta_rad), 0, 0) / np.where(dt==0, np.nan, dt)
    yaw_rate_deg = np.degrees(yaw_rate_rad)
    yaw_rate_rad[np.abs(yaw_rate_rad) > 0.2] = np.nan
    yaw_rate_deg[np.abs(yaw_rate_deg) > 10 ] = np.nan
    keep_yaw = ~np.isnan(yaw_rate_rad)
    if keep_yaw.sum() < 4: continue
    data,dev_c = data.loc[keep_yaw].reset_index(drop=True), dev_c[keep_yaw]
    dev_l,dev_r,dev_llc,dev_rlc = dev_l[keep_yaw],dev_r[keep_yaw],dev_llc[keep_yaw],dev_rlc[keep_yaw]
    theta_abs,theta_rad    = theta_abs[keep_yaw],theta_rad[keep_yaw]
    yaw_rate_deg,yaw_rate_rad = yaw_rate_deg[keep_yaw],yaw_rate_rad[keep_yaw]
    dt = dt[keep_yaw]; pts = data[["Y_Smt","X_Smt"]].to_numpy()
    near_x, near_y, tan_deg, tan_rad = [], [], [], []
    for p in pts:
        best=np.inf; chosen=None
        for a,b in zip(centerline_pts[:-1], centerline_pts[1:]):
            d,proj,ab = proj_distance(p, a, b)
            if d < best:
                best, chosen = d, (proj, ab)
        proj, ab = chosen
        near_x.append(proj[1]); near_y.append(proj[0])
        bearing_raw = math.degrees(math.atan2(ab[1], ab[0]))
        acute       = acute_angle(bearing_raw)
        tan_deg.append(acute)
        tan_rad.append(math.radians(acute))
    tan_deg = np.asarray(tan_deg); tan_rad = np.asarray(tan_rad)
    final_ang_deg = np.abs(theta_abs - tan_deg)
    final_ang_rad = np.radians(final_ang_deg)
    L_by_2sin = np.abs((lookup(LENGTH,vid)/2) * np.sin(final_ang_rad))
    speed = (data["Speed"].values if "Speed" in data
             else np.hypot(vx.iloc[keep_yaw].values, vy.iloc[keep_yaw].values))
    long_speed, lat_speed = speed*np.cos(final_ang_rad), speed*np.sin(final_ang_rad)
    delta_s    = np.hypot(data["X_Smt"].diff().fillna(0), data["Y_Smt"].diff().fillna(0))
    var_radius = radius_three_pts(data["X_Smt"], data["Y_Smt"])
    l,w = lookup(LENGTH,vid), lookup(WIDTH,vid)
    R   = var_radius.copy(); R[R==0] = np.nan
    widening_dyn = np.nan_to_num((l**2)/(2*R) + speed/(9.5*np.sqrt(R)), nan=0.0)
    est_widen    = np.abs(1.75 - np.abs(dev_c)) + np.abs(L_by_2sin)
    safety_var   = -(est_widen - widening_dyn)
    widening_const = (l**2)/(2*90) + 50/(9.5*np.sqrt(90))
    safety_codal   = -(est_widen - widening_const)
    safety_label   = np.where(safety_codal < 0, "Unsafe", "Safe")
    direction      = direction_of_travel(data["Y_Smt"])
    orig_lane_dist = dev_llc if direction=="Northbound" else dev_rlc
    total_time = (data["Time"].iloc[-1] - data["Time"].iloc[0]) if len(data) > 1 else 0
    ev_mask    = np.abs(yaw_rate_deg) >= 4
    evasive_rows.append({
        "VehicleID":vid,
        "TotalTime_s":round(total_time,3),
        "EvasiveTime_s":round(float(np.sum(dt[ev_mask])),3),
        "PositiveTime_s":round(float(np.sum(dt[yaw_rate_deg>=4])),3),
        "NegativeTime_s":round(float(np.sum(dt[yaw_rate_deg<=-4])),3),
        "PercentEvasive":round(100*np.sum(dt[ev_mask])/total_time,2) if total_time else 0
    })
    df_out = pd.DataFrame({
        "VehicleID":vid, "Time":data["Time"],
        "X":data.get("X", np.nan), "Y":data.get("Y", np.nan),
        "X_Smt":data["X_Smt"], "Y_Smt":data["Y_Smt"],
        "Long_Speed":data.get("Long_Speed", np.nan),
        "Lat_Speed" :data.get("Lat_Speed",  np.nan),
        "Speed":speed, "Long_Acc":data.get("Long_Acc", np.nan),
        "Lat_Acc":data.get("Lat_Acc",  np.nan),
        "Acceleration":data.get("Acceleration", np.nan),
        "Theta":data.get("Theta", np.nan), "ACT":data.get("ACT", np.nan),
        "ConflictVeh":data.get("ConflictVeh", np.nan),
        "ConflictType":data.get("ConflictType", np.nan), "ROR":data.get("ROR", np.nan),
        "Distance from centreline":dev_c,
        "Deviation":np.abs(1.75 - np.abs(dev_c)),
        "Distance from left edge":dev_l,
        "Left lateral margin":np.abs(w/2 - dev_l),
        "Distance from right edge":dev_r,
        "Right lateral margin":np.abs(w/2 - dev_r),
        "Distance from left lane centreline":dev_llc,
        "Distance from right lane centreline":dev_rlc,
        "Distance from original lane centreline":orig_lane_dist,
        "Position_Relative_to_Centerline":np.where(dev_c>0,"Right",
                                                   np.where(dev_c<0,"Left","Center")),
        "Direction":direction,
        "Heading_Angle_Deg":theta_abs, "Heading_Angle_Rad":theta_rad,
        "Yaw_Rate_Deg":yaw_rate_deg,  "Yaw_Rate_Rad":yaw_rate_rad,
        "L_by_2SinTheta":L_by_2sin,
        "Nearest_Centreline_X":near_x, "Nearest_Centreline_Y":near_y,
        "Curve_Angle_Rad":tan_rad,  "Curve_Angle_Deg":tan_deg,
        "Final_Angle_Rad":final_ang_rad, "Final_Angle_Deg":final_ang_deg,
        "Estimated_Curve_Widening":est_widen,
        "Longitudinal_Speed":long_speed, "Latitudinal_Speed":lat_speed,
        "DeltaS":delta_s, "Variable Radius":var_radius,
        "IRC_Extra_Widening_Dynamic":widening_dyn,
        "Safety_Margin_Variable":safety_var,
        "IRC_Extra_Widening_Constant":widening_const,
        "Safety_Margin_Codal":safety_codal,
        "Safety_Assessment":safety_label
    })
    rows_out.append(df_out)
    processed_ids.append(vid)
    plt.figure(figsize=(6,5))
    plt.plot(data["X_Smt"], data["Y_Smt"],
             lw=1.5, color=TYPE_COLOUR.get(vtype(vid),"tab:gray"))
    plt.plot(road_df["Left_E"],  road_df["Left_N"],  "g--",
             road_df["Right_E"], road_df["Right_N"], "r--",
             road_df["Center_E"],road_df["Center_N"],"k--")
    plt.axis("equal"); plt.grid(True); plt.title(f"Trajectory – {vid}")
    plt.tight_layout(); plt.savefig(os.path.join(vehtraj_dir, f"{vid}_traj.png")); plt.close()
    plt.figure(figsize=(6,3))
    plt.plot(data["Time"], yaw_rate_deg, color=TYPE_COLOUR.get(vtype(vid),"tab:gray"))
    plt.axhline( 4, color="r", ls="--"); plt.axhline(-4, color="r", ls="--")
    plt.ylim(-20,20); plt.grid(True)
    plt.title(f"Yaw rate (deg/s) – {vid}")
    plt.xlabel("Time (s)"); plt.ylabel("deg/s")
    plt.tight_layout(); plt.savefig(os.path.join(yawplot_dir, f"{vid}_yaw_deg.png")); plt.close()

final_df = pd.concat(rows_out, ignore_index=True)
main_cols = [
    "VehicleID","Time","X","Y","X_Smt","Y_Smt",
    "Long_Speed","Lat_Speed","Speed","Long_Acc","Lat_Acc","Acceleration",
    "Theta","ACT","ConflictVeh","ConflictType","ROR",
    "Distance from centreline","Deviation",
    "Distance from left edge","Left lateral margin",
    "Distance from right edge","Right lateral margin",
    "Distance from left lane centreline","Distance from right lane centreline",
    "Distance from original lane centreline",
    "Position_Relative_to_Centerline","Direction",
    "Heading_Angle_Deg","Heading_Angle_Rad","Yaw_Rate_Deg","Yaw_Rate_Rad",
    "L_by_2SinTheta","Nearest_Centreline_X","Nearest_Centreline_Y",
    "Curve_Angle_Rad","Curve_Angle_Deg","Final_Angle_Rad","Final_Angle_Deg",
    "Estimated_Curve_Widening","Longitudinal_Speed","Latitudinal_Speed",
    "DeltaS","Variable Radius",
    "IRC_Extra_Widening_Dynamic","Safety_Margin_Variable",
    "IRC_Extra_Widening_Constant","Safety_Margin_Codal","Safety_Assessment"
]
for c in main_cols:
    if c not in final_df: final_df[c] = np.nan
final_df = final_df[main_cols]
final_df.to_csv(os.path.join(downloads, "Final_Merged_Trajectory.csv"), index=False)
pd.DataFrame(evasive_rows).to_csv(os.path.join(downloads, "Evasive_Time_Summary.csv"), index=False)

def steering_entropy_3rd(ang):
    if len(ang) < 5: return np.nan
    errs=[]
    for i in range(4, len(ang)):
        pred = (ang[i-1] + (ang[i-1]-ang[i-2])
                + 0.5*((ang[i-1]-ang[i-2])-(ang[i-2]-ang[i-3]))
                + (1/6)*((ang[i-1]-3*ang[i-2]+3*ang[i-3]-ang[i-4])))
        errs.append(ang[i] - pred)
    alpha = np.percentile(np.abs(errs), 90)
    if alpha == 0: return 0
    bins = np.linspace(-alpha, alpha, 10)
    hist,_ = np.histogram(errs, bins=bins)
    p = hist / np.sum(hist); p = p[p>0]
    return -np.sum(p*np.log(p)/np.log(9))

def sd_and_range(ang):
    if len(ang) < 2: return np.nan, np.nan
    return np.std(ang), np.max(ang) - np.min(ang)

entropy_rows=[]
for veh, grp in final_df.groupby("VehicleID"):
    a = grp["Heading_Angle_Rad"].astype(float).values
    entropy_rows.append({
        "VehicleID":veh,
        "Entropy_3rdOrder":steering_entropy_3rd(a),
        "SDSA":sd_and_range(a)[0],
        "SAR" :sd_and_range(a)[1],
        "Std_Heading_Angle":np.std(a)
    })
pd.DataFrame(entropy_rows).to_csv(os.path.join(downloads, "Steering_Metrics_Final_Cleaned.csv"), index=False)

shutil.make_archive(os.path.join(downloads, "Vehicle_Trajectories"),"zip",vehtraj_dir)
shutil.make_archive(os.path.join(downloads, "YawRate_Plots"),"zip",yawplot_dir)
for vt in ["Car","Bike","LCV","Truck"]:
    ids = [v for v in processed_ids if vtype(v) == vt]
    if not ids: continue
    plt.figure(figsize=(7,6))
    plt.plot(road_df["Left_E"], road_df["Left_N"],"g--", label="Left edge")
    plt.plot(road_df["Right_E"],road_df["Right_N"],"r--", label="Right edge")
    plt.plot(road_df["Center_E"],road_df["Center_N"],"k--", label="Centre-line")
    for v in ids:
        sub = vehicle_df[vehicle_df["VehicleID"]==v]
        plt.plot(sub["X_Smt"], sub["Y_Smt"], lw=1.0,
                 color=TYPE_COLOUR.get(vt,"tab:gray"), label=v)
    plt.axis("equal"); plt.grid(True)
    plt.title(f"Projected Trajectories – {vt}s")
    plt.legend(fontsize=7, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(typeproj_dir, f"{vt}_projections.png")); plt.close()
shutil.make_archive(os.path.join(downloads, "VehicleType_Plots"),"zip",typeproj_dir)

plt.figure(figsize=(7,6))
plt.plot(road_df["Left_E"], road_df["Left_N"],"g--", label="Left edge")
plt.plot(road_df["Right_E"],road_df["Right_N"],"r--", label="Right edge")
plt.plot(road_df["Center_E"],road_df["Center_N"],"k--", label="Centre-line")
for v in processed_ids:
    c = TYPE_COLOUR.get(vtype(v),"tab:gray")
    sub = vehicle_df[vehicle_df["VehicleID"]==v]
    plt.plot(sub["X_Smt"], sub["Y_Smt"], lw=.7, alpha=.5, color=c, label=vtype(v))
h,l = plt.gca().get_legend_handles_labels(); uniq={}
for hh,ll in zip(h,l):
    if ll not in uniq: uniq[ll]=hh
plt.legend(uniq.values(), uniq.keys(), frameon=False)
plt.axis("equal"); plt.grid(True)
plt.title("All Vehicle Trajectories (raw)")
plt.tight_layout(); plt.savefig(os.path.join(downloads, "All_Vehicle_Trajectories.png")); plt.close()
print("All outputs saved to", downloads)
