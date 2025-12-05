import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

csv_path = r"FINAL SET/sweep_rev_const_density_results_19112025_1649.csv"
out_png = r"FINAL SET/sweep_rev_const_density_plot_19112025_1649_partial.png"

print(f"Reading: {csv_path}")
df = pd.read_csv(csv_path)
# ensure numeric
for c in ['dir_D_eff','neu_D_eff','porosity','tau_dir','tau_neu']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.dropna(subset=['dir_D_eff','neu_D_eff'])
if df.empty:
    print('No data available to plot.')
    sys.exit(1)

grouped = df.groupby('L_m').agg(['mean','std','count'])
Ls = np.array(grouped.index, dtype=float)
# Dirichlet
dir_mean = grouped['dir_D_eff']['mean'].values
dir_se = (grouped['dir_D_eff']['std'] / np.sqrt(grouped['dir_D_eff']['count'])).values
# Neumann
neu_mean = grouped['neu_D_eff']['mean'].values
neu_se = (grouped['neu_D_eff']['std'] / np.sqrt(grouped['neu_D_eff']['count'])).values

plt.figure(figsize=(6,4))
plt.errorbar(Ls, dir_mean, yerr=dir_se, marker='o', linestyle='-', label='Dirichlet (mean ± SE)')
plt.errorbar(Ls, neu_mean, yerr=neu_se, marker='s', linestyle='--', label='Neumann (mean ± SE)')
plt.xscale('log')
plt.xlabel('Domain size L (m)')
plt.ylabel('D_eff (m^2/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_png, dpi=200)
print(f'Saved plot to {out_png}')

# Print summary table
print('\nSummary by L (mean ± SE):')
for i,L in enumerate(Ls):
    print(f"L={L:.3e}: Dir={dir_mean[i]:.3e} ± {dir_se[i]:.3e}, Neu={neu_mean[i]:.3e} ± {neu_se[i]:.3e}, n={int(grouped['dir_D_eff']['count'].iloc[i])}")
