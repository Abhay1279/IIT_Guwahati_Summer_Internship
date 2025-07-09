import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.stats import norm
from scipy.optimize import fsolve
from google.colab import files
import io

uploaded = files.upload()
file_names = list(uploaded.keys())

def compute_heading_angle(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    theta = np.arctan2(dy, dx)
    theta = np.insert(theta, 0, theta[0])
    return theta

def compute_steering_entropy(angles):
    if len(angles) < 5:
        return np.nan
    errors = []
    for i in range(4, len(angles)):
        theta_p = (
            angles[i-1]
            + (angles[i-1] - angles[i-2])
            + 0.5*((angles[i-1] - angles[i-2]) - (angles[i-2] - angles[i-3]))
            + (1/6)*((angles[i-1] - 3*angles[i-2] + 3*angles[i-3] - angles[i-4]))
        )
        e = angles[i] - theta_p
        errors.append(e)
    alpha = np.percentile(np.abs(errors), 90)
    if alpha == 0:
        return 0
    bins = np.linspace(-alpha, alpha, 10)
    hist, _ = np.histogram(errors, bins=bins)
    p = hist / np.sum(hist)
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p) / np.log(9))
    return entropy

all_entropies = []
file_labels = []
vehicle_ids = []

for fname in file_names:
    print(f"Processing: {fname}")
    df = pd.read_csv(io.BytesIO(uploaded[fname]))
    for veh_id, group in df.groupby("VehicleID"):
        group_sorted = group.sort_values("Time")
        x = group_sorted["X_Smt"].values
        y = group_sorted["Y_Smt"].values
        headings = compute_heading_angle(x, y)
        entropy = compute_steering_entropy(headings)
        all_entropies.append(entropy)
        file_labels.append(fname)
        vehicle_ids.append(veh_id)

entropy_df = pd.DataFrame({
    "VehicleID": vehicle_ids,
    "Entropy": all_entropies,
    "File": file_labels
})

valid_entropies = entropy_df["Entropy"].dropna().values.reshape(-1,1)

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(valid_entropies)

labels_gmm = gmm.predict(valid_entropies)
probs_gmm = gmm.predict_proba(valid_entropies)

sorted_idx = np.argsort(gmm.means_.flatten())
weights = gmm.weights_[sorted_idx]
means = gmm.means_.flatten()[sorted_idx]
variances = gmm.covariances_.flatten()[sorted_idx]
stds = np.sqrt(variances)

x_plot = np.linspace(valid_entropies.min()-0.1, valid_entropies.max()+0.1, 1000)
component_densities = []
for k in range(3):
    density = weights[k]*norm.pdf(x_plot, loc=means[k], scale=stds[k])
    component_densities.append(density)

def find_intersection(k):
    def diff(x):
        return (
            weights[k]*norm.pdf(x, means[k], stds[k]) -
            weights[k+1]*norm.pdf(x, means[k+1], stds[k+1])
        )
    x0 = (means[k] + means[k+1])/2
    return fsolve(diff, x0)[0]

threshold1 = find_intersection(0)
threshold2 = find_intersection(1)

print("✅ Thresholds from Gaussian intersections:")
print(f"Normal/Distracted: {threshold1:.4f}")
print(f"Distracted/Erratic: {threshold2:.4f}")

log_likelihood = gmm.score(valid_entropies) * len(valid_entropies)
aic = gmm.aic(valid_entropies)
bic = gmm.bic(valid_entropies)
converged = gmm.converged_
n_iter = gmm.n_iter_
avg_log_likelihood = gmm.score(valid_entropies)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(valid_entropies)
labels_kmeans = kmeans.labels_
centers_kmeans = kmeans.cluster_centers_.flatten()
sorted_centers = np.sort(centers_kmeans)
threshold1_kmeans = (sorted_centers[0] + sorted_centers[1])/2
threshold2_kmeans = (sorted_centers[1] + sorted_centers[2])/2
inertia = kmeans.inertia_
kmeans_n_iter = kmeans.n_iter_

def assign_category_gmm(entropy):
    if pd.isna(entropy):
        return "Insufficient Data"
    elif entropy <= threshold1:
        return "Normal"
    elif entropy <= threshold2:
        return "Distracted"
    else:
        return "Erratic"

def assign_category_kmeans(entropy):
    if pd.isna(entropy):
        return "Insufficient Data"
    elif entropy <= threshold1_kmeans:
        return "Normal"
    elif entropy <= threshold2_kmeans:
        return "Distracted"
    else:
        return "Erratic"

entropy_df["Category_GMM"] = entropy_df["Entropy"].apply(assign_category_gmm)
entropy_df["Category_KMeans"] = entropy_df["Entropy"].apply(assign_category_kmeans)

prob_cols = [f"GMM_Prob_Cluster{i+1}" for i in range(3)]
probs_df = pd.DataFrame(probs_gmm, columns=prob_cols)
probs_df.index = entropy_df[entropy_df["Entropy"].notna()].index
entropy_df = entropy_df.join(probs_df)
plt.figure(figsize=(12,6))
sns.kdeplot(valid_entropies.flatten(), fill=True, color="gray", label="Empirical Density")
colors = ["red", "blue", "green"]
for k in range(3):
    plt.plot(
        x_plot,
        component_densities[k],
        color=colors[k],
        linestyle="--",
        label=f"Component {k+1}"
    )
plt.plot(x_plot, np.sum(component_densities, axis=0), color="black", linewidth=2, label="Mixture")
plt.axvline(threshold1, color="purple", linestyle="-", label="Threshold 1 (Intersection)")
plt.axvline(threshold2, color="purple", linestyle="-", label="Threshold 2 (Intersection)")
plt.xlabel("Entropy")
plt.ylabel("Density")
plt.title("GMM Components and Intersections as Thresholds")
plt.legend()
plot_filename = "Entropy_GMM_Distribution.png"
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
plt.show()
files.download(plot_filename)

print("\n✅ GMM Metrics:")
print("Means:", means)
print("Variances:", variances)
print("Weights:", weights)
print("Log Likelihood:", log_likelihood)
print("AIC:", aic)
print("BIC:", bic)
print("Converged:", converged)
print("EM Iterations:", n_iter)
print("Threshold1:", threshold1)
print("Threshold2:", threshold2)
print("\n✅ K-Means Metrics:")
print("Centers:", sorted_centers)
print("Inertia:", inertia)
print("Iterations:", kmeans_n_iter)
print("Threshold1:", threshold1_kmeans)
print("Threshold2:", threshold2_kmeans)

entropy_filename = "Steering_Entropy_Categories_WithThresholds.csv"
entropy_df.to_csv(entropy_filename, index=False)
files.download(entropy_filename)

metrics = {
    "GMM_Means": means,
    "GMM_Variances": variances,
    "GMM_Weights": weights,
    "GMM_LogLikelihood": [log_likelihood],
    "GMM_AIC": [aic],
    "GMM_BIC": [bic],
    "GMM_Converged": [converged],
    "GMM_Iterations": [n_iter],
    "GMM_Threshold1": [threshold1],
    "GMM_Threshold2": [threshold2],
    "KMeans_Centers": sorted_centers,
    "KMeans_Inertia": [inertia],
    "KMeans_Iterations": [kmeans_n_iter],
    "KMeans_Threshold1": [threshold1_kmeans],
    "KMeans_Threshold2": [threshold2_kmeans]
}
metrics_df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in metrics.items()]))
metrics_filename = "Model_Metrics_Summary.csv"
metrics_df.to_csv(metrics_filename, index=False)
files.download(metrics_filename)
