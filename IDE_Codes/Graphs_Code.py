import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
import os

plt.rcParams["figure.dpi"] = 200

file_path = input("Enter path to Final_Merged_Trajectory*.csv: ").strip()
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip().str.replace(' ', '_', regex=False)
df["VehicleType"] = df["VehicleID"].astype(str).str.extract(r"([A-Za-z]+)")

results = []
for vid, g in df.groupby("VehicleID"):
    asa = np.mean(np.abs(g["Heading_Angle_Rad"]))
    sdsa = np.std(g["Heading_Angle_Rad"])
    sar = g["Heading_Angle_Rad"].max() - g["Heading_Angle_Rad"].min()
    angle_diff = g["Heading_Angle_Rad"].diff().fillna(0)
    srr = (np.abs(angle_diff) > 0.1).sum()
    dist = np.abs(g["Distance_from_centreline"])
    mlp = dist.mean()
    sdlp = dist.std()
    lpr = dist.max() - dist.min()
    out_lane_dur = ((dist > 1.675).sum()) * g["Time"].diff().mean()
    results.append({
        "VehicleID": vid,
        "VehicleType": g["VehicleType"].iloc[0],
        "ASA": asa, "SDSA": sdsa, "SAR": sar, "SRR": srr,
        "MLP": mlp, "SDLP": sdlp, "LPR": lpr,
        "OutOfLaneDuration": out_lane_dur
    })
metrics_df = pd.DataFrame(results)

def steering_entropy_3rd(angles):
    if len(angles) < 5: return np.nan
    errs = []
    for i in range(4, len(angles)):
        θp = (angles[i-1] +
              (angles[i-1] - angles[i-2]) +
              0.5 * ((angles[i-1] - angles[i-2]) - (angles[i-2] - angles[i-3])) +
              (1/6) * (angles[i-1] - 3*angles[i-2] + 3*angles[i-3] - angles[i-4]))
        errs.append(angles[i] - θp)
    α = np.percentile(np.abs(errs), 90)
    if α == 0: return 0
    hist,_ = np.histogram(errs, bins=np.linspace(-α, α, 10))
    p = hist/ hist.sum()
    p = p[p>0]
    return -np.sum(p * np.log(p) / np.log(9))

entropy = [{"VehicleID": vid,
            "Entropy_3rdOrder": steering_entropy_3rd(g["Heading_Angle_Rad"].values)}
           for vid,g in df.groupby("VehicleID")]
entropy_df = pd.DataFrame(entropy)
metrics_df = metrics_df.merge(entropy_df, on="VehicleID")
df = df.merge(metrics_df, on="VehicleID", suffixes=("", "_metric"))

folders = {
    "bar":            "BarPlots",
    "scatter_avg":    "ScatterPlots_Averages",
    "veh_type_plots": "VehicleType_Distribution_Plots"
}
for path in folders.values(): os.makedirs(path, exist_ok=True)

metrics = ["ASA","SDSA","SAR","SRR","MLP","SDLP","LPR","OutOfLaneDuration","Entropy_3rdOrder"]
for m in metrics:
    plt.figure(figsize=(16,8))
    vals = metrics_df[m]
    plt.bar(metrics_df["VehicleID"].astype(str), vals,
            color="#69b3a2", edgecolor="black")
    plt.ylabel(m, fontsize=16)
    plt.xlabel("Vehicle ID", fontsize=16)
    plt.title(f"{m} per Vehicle", fontsize=18, weight="bold")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, vals.max()*1.2)
    plt.tight_layout()
    plt.savefig(f"{folders['bar']}/{m}_BarPlot.png", dpi=300)
    plt.close()

pairs = [("ASA","SDSA"), ("SRR","MLP"), ("OutOfLaneDuration","SAR"), ("LPR","SDLP")]
for x,y in pairs:
    plt.figure(figsize=(10,8))
    plt.scatter(metrics_df[x], metrics_df[y],
                s=80, alpha=0.8, color="#4682B4", edgecolors="black")
    m,b = np.polyfit(metrics_df[x], metrics_df[y], 1)
    plt.plot(metrics_df[x], m*metrics_df[x]+b,
             linestyle="--", linewidth=2, color="red")
    plt.xlabel(x, fontsize=16);  plt.ylabel(y, fontsize=16)
    plt.title(f"{y} vs {x} (per vehicle)", fontsize=18, weight="bold")
    plt.grid(True, linestyle='--', alpha=0.6);  plt.tight_layout()
    plt.savefig(f"{folders['scatter_avg']}/{y}_vs_{x}_Scatter.png", dpi=300)
    plt.close()

corr = df[metrics].corr()
mask_lower = corr.copy()
for i in range(len(metrics)):
    for j in range(len(metrics)):
        if i < j: mask_lower.iat[i,j] = ""
mask_lower.to_csv("Table_2_CorrelationMatrix.csv")

from sklearn.linear_model import LinearRegression
vif = pd.DataFrame(index=metrics, columns=metrics)
for target in metrics:
    X = df[[m for m in metrics if m != target]].dropna().reset_index(drop=True)
    y = df[target].dropna().reset_index(drop=True)
    n = min(len(X), len(y))
    X,y = X.iloc[:n], y.iloc[:n]
    r2  = LinearRegression().fit(X,y).score(X,y)
    vif_val = 1/(1-r2) if r2 < 1 else np.inf
    vif.loc[target,:] = ""
    vif.loc[target,target] = "N/A"
    for pred in metrics:
        if pred != target: vif.loc[target,pred] = f"{vif_val:.2f}"
vif.to_csv("Table_4_GeneralizedVIF.csv")

veh_types = df["VehicleType"].unique()
plt.figure(figsize=(10,6))
data = [df[df["VehicleType"]==vt]["Distance_from_centreline"].dropna() for vt in veh_types]
plt.boxplot(data, labels=veh_types, patch_artist=True,
            medianprops=dict(color="black"),
            boxprops=dict(facecolor="#69b3a2", color="black"),
            whiskerprops=dict(color="black"), capprops=dict(color="black"),
            flierprops=dict(markerfacecolor="red", marker="o", markersize=4, alpha=0.6))
plt.ylabel("Deviation from Centreline (m)", fontsize=14)
plt.xlabel("Vehicle Type", fontsize=14)
plt.title("Deviation Distribution by Vehicle Type", fontsize=16, weight="bold")
plt.grid(axis='y', linestyle='--', alpha=0.6);  plt.tight_layout()
plt.savefig(f"{folders['veh_type_plots']}/Deviation_Boxplot_By_VehicleType.png", dpi=300)
plt.close()

plt.figure(figsize=(10,6))
cw_data = [df[df["VehicleType"]==vt]["Estimated_Curve_Widening"].dropna() for vt in veh_types]
plt.boxplot(cw_data, labels=veh_types, patch_artist=True,
            medianprops=dict(color="black"),
            boxprops=dict(facecolor="#ff7f0e", color="black"),
            whiskerprops=dict(color="black"), capprops=dict(color="black"),
            flierprops=dict(markerfacecolor="red", marker="o", markersize=4, alpha=0.6))
plt.ylabel("Estimated Curve Widening (m)", fontsize=14)
plt.xlabel("Vehicle Type", fontsize=14)
plt.title("Estimated Curve Widening by Vehicle Type", fontsize=16, weight="bold")
plt.grid(axis='y', linestyle='--', alpha=0.6);  plt.tight_layout()
plt.savefig(f"{folders['veh_type_plots']}/EstimatedCurveWidening_Boxplot_By_VehicleType.png", dpi=300)
plt.close()

zip_name = "Driving_Metrics_AllOutputs.zip"
with ZipFile(zip_name, 'w') as z:
    for root,_,files_ in os.walk('.'):
        for f in files_:
            if f.endswith(('.png','.csv')) and not f.startswith('.'):
                z.write(os.path.join(root,f))
print("✓ Done! All outputs zipped to", zip_name)
