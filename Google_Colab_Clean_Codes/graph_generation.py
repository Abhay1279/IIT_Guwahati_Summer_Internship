import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
import io

uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[file_name]))

if 'VehicleID' not in df.columns or 'Heading_Angle_Rad' not in df.columns:
    raise ValueError("CSV must contain 'VehicleID' and 'Heading_Angle_Rad' columns.")

all_errors = []
for veh_id, group in df.groupby('VehicleID'):
    angles = group['Heading_Angle_Rad'].values
    if len(angles) >= 5:
        for i in range(4, len(angles)):
            theta_p = (
                angles[i-1]
                + (angles[i-1] - angles[i-2])
                + 0.5 * ((angles[i-1] - angles[i-2]) - (angles[i-2] - angles[i-3]))
                + (1/6) * ((angles[i-1] - 3*angles[i-2] + 3*angles[i-3] - angles[i-4]))
            )
            e = angles[i] - theta_p
            all_errors.append(e)

all_errors = np.array(all_errors)

if len(all_errors) == 0:
    print("No prediction errors to plot.")
else:
    alpha = np.percentile(np.abs(all_errors), 90)
    if alpha < 1e-5:
        print("Alpha too small. Forcing alpha to 0.01 for visibility.")
        alpha = 0.01
    bin_edges = np.linspace(-5*alpha, 5*alpha, 10)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_counts, _ = np.histogram(all_errors, bins=bin_edges)
    proportions = hist_counts / np.sum(hist_counts)
    plt.figure(figsize=(10,6))
    sns.kdeplot(all_errors, bw_adjust=1.0, fill=True, color='gray', alpha=0.3, label='KDE')
    plt.hist(all_errors, bins=bin_edges, color='skyblue', edgecolor='black', alpha=0.6, label='Histogram')

    for edge in bin_edges:
        plt.axvline(x=edge, color='k', linestyle='--', linewidth=0.8)

    for i, center in enumerate(bin_centers):
        plt.text(center, 0.005, f'P{i+1}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xlabel("Prediction Error e(n)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors into 9 Bins (All Vehicles)")
    plt.ylim(0, 100)  
    plt.legend()
    plt.tight_layout()
    plot_filename = "Prediction_Error_Distribution.png"
    plt.savefig(plot_filename, dpi=300)
    files.download(plot_filename)
    plt.show()
