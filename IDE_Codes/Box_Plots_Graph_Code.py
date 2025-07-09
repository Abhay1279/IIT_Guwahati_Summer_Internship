import pandas as pd, numpy as np, matplotlib.pyplot as plt, io, os, zipfile
from tqdm import tqdm

csv_path = input("Enter path to merged CSV: ").strip()
df = pd.read_csv(csv_path)
print(f'✅ Loaded {csv_path}  ({len(df):,} rows)')

print("\nColumns in the file:")
for i, c in enumerate(df.columns): print(f'  {i:>2}: {c}')
section_col = input('\n👉  Type the exact column name that stores Tangent / Transition / Circular labels → ').strip()
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

out_dir = "Boxplot_Images"
os.makedirs(out_dir, exist_ok=True)
img_files = []

for col in tqdm(METRICS, desc='Plotting'):
    data = [df.loc[df['Section'] == cat, col].dropna() for cat in present]
    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=present, showfliers=False)
    plt.title(f'{col} by Section')
    plt.ylabel(col)
    plt.tight_layout()
    img_name = f'{col.replace(" ", "_")}_box.png'
    plt.savefig(os.path.join(out_dir, img_name), format='png', dpi=300)
    plt.close()
    img_files.append(os.path.join(out_dir, img_name))

zip_name = '12_Boxplots.zip'
with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for img in img_files:
        zf.write(img, os.path.basename(img))

print('\n✓ Done! 12 plots zipped as', zip_name)
