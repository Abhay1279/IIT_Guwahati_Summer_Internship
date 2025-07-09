import pandas as pd
import numpy as np
from google.colab import files
import io

uploaded = files.upload()
file_name = next(iter(uploaded))
df = pd.read_csv(io.BytesIO(uploaded[file_name]))

def classify_lane_status(group):
    signs = np.sign(group['Distance from centreline'])
    return 'Encroaching' if (signs != signs.iloc[0]).any() else 'In Lane'

encroachment_status = df.groupby('VehicleID').apply(classify_lane_status).reset_index()
encroachment_status.columns = ['VehicleID', 'Encroachment_Status']

agg_columns = {
    'Deviation': ['mean', 'max', 'min', 'std'],
    'Safety_Margin_Variable': ['mean', 'max', 'min', 'std'],
    'Safety_Margin_Codal': ['mean', 'max', 'min', 'std'],
    'IRC_Extra_Widening_Dynamic': ['mean', 'max'],
    'IRC_Extra_Widening_Constant': ['mean', 'max'],
    'Estimated curve widening': ['mean', 'max'],
    'L_by_2SinTheta': ['mean', 'max'],
    'DeltaS': 'mean',
    'Variable Radius': 'mean',
    'Speed': 'mean',
    'Direction': 'first',
}

grouped = df.groupby('VehicleID').agg(agg_columns)

grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

grouped.rename(columns={
    'Deviation_std': 'Deviation_standard_deviation',
    'Safety_Margin_Variable_std': 'Safety_Margin_Variable_standard_deviation',
    'Safety_Margin_Codal_std': 'Safety_Margin_Codal_standard_deviation',
}, inplace=True)

grouped['Deviation_coefficient_of_variation'] = grouped['Deviation_standard_deviation'] / grouped['Deviation_mean']
grouped['Safety_Margin_Variable_coefficient_of_variation'] = grouped['Safety_Margin_Variable_standard_deviation'] / grouped['Safety_Margin_Variable_mean']
grouped['Safety_Margin_Codal_coefficient_of_variation'] = grouped['Safety_Margin_Codal_standard_deviation'] / grouped['Safety_Margin_Codal_mean']

grouped.reset_index(inplace=True)
grouped.rename(columns={'Direction_first': 'Direction'}, inplace=True)
grouped = grouped.merge(encroachment_status, on='VehicleID', how='left')
ordered_cols = [
    'VehicleID',
    'DeltaS_mean', 'Variable Radius_mean',
    'Direction',
    'Encroachment_Status',
    'Speed_mean',
    'Deviation_mean', 'Deviation_max', 'Deviation_min', 'Deviation_standard_deviation', 'Deviation_coefficient_of_variation',
    'Safety_Margin_Variable_mean', 'Safety_Margin_Variable_max', 'Safety_Margin_Variable_min',
    'Safety_Margin_Variable_standard_deviation', 'Safety_Margin_Variable_coefficient_of_variation',
    'Safety_Margin_Codal_mean', 'Safety_Margin_Codal_max', 'Safety_Margin_Codal_min',
    'Safety_Margin_Codal_standard_deviation', 'Safety_Margin_Codal_coefficient_of_variation',
    'IRC_Extra_Widening_Dynamic_mean', 'IRC_Extra_Widening_Dynamic_max',
    'IRC_Extra_Widening_Constant_mean', 'IRC_Extra_Widening_Constant_max',
    'Estimated curve widening_mean', 'Estimated curve widening_max',
    'L_by_2SinTheta_mean', 'L_by_2SinTheta_max'
]

ordered_cols = [col for col in ordered_cols if col in grouped.columns]
grouped = grouped[ordered_cols]
output_file = "Final_Vehicle_Summary_With_Encroachment.csv"
grouped.to_csv(output_file, index=False)
files.download(output_file)
