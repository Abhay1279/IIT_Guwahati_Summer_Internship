import pandas as pd, numpy as np, matplotlib.pyplot as plt, io, os, shutil, zipfile
from tqdm import tqdm
from google.colab import files

print("📤 Upload your merged CSV …")
up       = files.upload()
csv_name = next(iter(up))
df       = pd.read_csv(io.BytesIO(up[csv_name]))
print(f'✅ Loaded {csv_name}  ({len(df):,} rows)')
print("\nColumns in the file:")
for i, c in enumerate(df.columns): print(f'  {i:>2}: {c}')
section_col = input(
    '\n👉  Type the exact column name that stores Tangent / Transition / Circular labels → '
).strip()
if section_col not in df.columns:
    raise ValueError(f'“{section_col}” is not a column in the file.')
def norm(lbl):
    if pd.isna(lbl): return np.nan
    l = str(lbl).lower()
    if 'tan'  in l or 'straig' in l:  return 'Tangent'
    if 'trans' in l:                  return 'Transition'
    if 'circ' in l or 'curve' in l:   return 'Circular'
    return np.nan
df['Section'] = df[section_col].apply(norm)
df = df.dropna(subset=['Section'])
print(f'🧮  Rows kept after label normalisation: {len(df):,}')
METRICS = [
    'Distance from centreline', 'Deviation',
    'Distance from left edge', 'Left lateral margin',
    'Distance from right edge', 'Right lateral margin',
    'Distance from original lane centreline', 'Distance from right lane centreline',
    'Estimated_Curve_Widening',
    'IRC_Extra_Widening_Dynamic', 'Safety_Margin_Variable', 'Safety_Margin_Codal'
]
missing = [m for m in METRICS if m not in df.columns]
if missing:
    raise ValueError(f'These columns are absent in the CSV:\n{missing}')
order   = ['Tangent', 'Transition', 'Circular']
present = [o for o in order if o in df['Section'].unique()]
zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
    for col in tqdm(METRICS, desc='Plotting'):
        data = [df.loc[df['Section'] == cat, col].dropna() for cat in present]
        plt.figure(figsize=(6, 4))
        plt.boxplot(data, labels=present, showfliers=False)
        plt.title(f'{col} by Section')
        plt.ylabel(col)
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=300)
        plt.close()
        zf.writestr(f'{col.replace(" ", "_")}_box.png', img.getvalue())
zip_buf.seek(0)
out_name = '12_Boxplots.zip'
with open(out_name, 'wb') as f: f.write(zip_buf.read())
files.download(out_name)
print('\n✅ Done! 12 plots zipped and download should start automatically.')
