import pandas as pd
from google.colab import files
import io

print("🔹 Please upload your behavior classification CSV file (the one with 'Behavior Class' column):")
uploaded = files.upload()

filename = list(uploaded.keys())[0]

df = pd.read_csv(io.BytesIO(uploaded[filename]))

print("\n🔹 Preview of your data:")
display(df.head())

if 'Behavior Class' not in df.columns:
    raise ValueError("❌ The file does not contain a 'Behavior Class' column. Please check your CSV.")

counts = df['Behavior Class'].value_counts()
total = counts.sum()
percentages = (counts / total) * 100
summary_df = pd.DataFrame({
    'Count of Vehicles': counts,
    'Percentage (%)': percentages.round(2)
}).reset_index().rename(columns={'index': 'Behavior Class'})

print("\n✅ Summary of Behavior Classification:")
display(summary_df)
print("\n🔹 Plain text output:")
for i, row in summary_df.iterrows():
    print(f"{row['Behavior Class']}: {row['Count of Vehicles']} vehicles ({row['Percentage (%)']}%)")
summary_df.to_csv('Behavior_Class_Summary.csv', index=False)
print("\n✅ Summary CSV saved as 'Behavior_Class_Summary.csv'")
