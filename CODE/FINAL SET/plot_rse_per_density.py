import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Map density labels to their sweep files (same as other analysis scripts)
DENSITY_FILES = {
    1e9: Path('const_density_1e09') / 'sweep_const_density_1e09_full.csv',
    1e12: Path('const_density_1e12') / 'sweep_const_density_1e12_full.csv',
    1e13: Path('const_density_1e13') / 'sweep_const_density_1e13_full.csv',
    1e14: Path('const_density_1e14') / 'sweep_const_density_1e14_full.csv',
}

OUT_TEMPLATE = 'rse_rev_{:.0e}.png'


def make_rse_plot(rho, csv_path):
    if not csv_path.is_file():
        print(f'Warning: file not found for density {rho:.0e}: {csv_path}')
        return

    df = pd.read_csv(csv_path)

    if 'L_m' not in df.columns:
        print(f"No 'L_m' column in {csv_path}; skipping.")
        return

    # Filter out failed runs if there is an error column
    if 'error' in df.columns:
        df = df[df['error'].isna() | (df['error'] == '')]

    # Try common column names for D_eff
    dir_cols = ['dir_D_eff', 'D_eff_dirichlet', 'Deff_dir', 'Deff_D']
    neu_cols = ['neu_D_eff', 'D_eff_neumann', 'Deff_neu', 'Deff_N']

    dir_col = next((c for c in dir_cols if c in df.columns), None)
    neu_col = next((c for c in neu_cols if c in df.columns), None)

    if dir_col is None or neu_col is None:
        print(f'Could not find D_eff columns in {csv_path}. Columns: {df.columns.tolist()}')
        return

    # Group by L and compute mean, std, N
    grouped = df.groupby('L_m').agg(
        dir_mean=(dir_col, 'mean'),
        dir_std=(dir_col, 'std'),
        dir_n=(dir_col, 'count'),
        neu_mean=(neu_col, 'mean'),
        neu_std=(neu_col, 'std'),
        neu_n=(neu_col, 'count'),
    ).reset_index()

    L = grouped['L_m'].to_numpy()

    # Avoid division by zero
    dir_mean = grouped['dir_mean'].to_numpy()
    dir_std = grouped['dir_std'].fillna(0.0).to_numpy()
    dir_n = grouped['dir_n'].to_numpy().astype(float)

    neu_mean = grouped['neu_mean'].to_numpy()
    neu_std = grouped['neu_std'].fillna(0.0).to_numpy()
    neu_n = grouped['neu_n'].to_numpy().astype(float)

    # Relative Standard Error (RSE = std / (mean * sqrt(N)))
    dir_rse = np.zeros_like(dir_mean)
    neu_rse = np.zeros_like(neu_mean)

    valid_dir = (dir_mean != 0) & (dir_n > 0)
    valid_neu = (neu_mean != 0) & (neu_n > 0)

    dir_rse[valid_dir] = dir_std[valid_dir] / (np.abs(dir_mean[valid_dir]) * np.sqrt(dir_n[valid_dir]))
    neu_rse[valid_neu] = neu_std[valid_neu] / (np.abs(neu_mean[valid_neu]) * np.sqrt(neu_n[valid_neu]))

    plt.figure(figsize=(8, 5))

    # Background shaded bands for RSE quality ranges
    rse_min, rse_max = 1e-3, 1.0
    L_min, L_max = L.min(), L.max()
    # Very good: <5%
    plt.fill_between([L_min, L_max], 1e-3, 0.05, color='green', alpha=0.08, label='< 5%')
    # Good: 5-10%
    plt.fill_between([L_min, L_max], 0.05, 0.10, color='yellow', alpha=0.08, label='5–10%')
    # Moderate: 10-20%
    plt.fill_between([L_min, L_max], 0.10, 0.20, color='orange', alpha=0.08, label='10–20%')
    # Poor: >20%
    plt.fill_between([L_min, L_max], 0.20, rse_max, color='red', alpha=0.05, label='> 20%')

    plt.loglog(L, dir_rse, '-o', label='Dirichlet RSE', color='tab:blue')
    plt.loglog(L, neu_rse, '-s', label='Neumann RSE', color='tab:purple')
    plt.xlabel('Domain size L [m]')
    plt.ylabel('Relative standard error (RSE)')
    plt.title(f'RSE of D_eff vs L at density {rho:.0e} pores/m³')
    plt.ylim(rse_min, rse_max)
    plt.grid(True, which='both', linestyle=':', alpha=0.4)
    # Put legend outside so it doesn't block curves
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()

    out_name = OUT_TEMPLATE.format(rho)
    plt.savefig(out_name, dpi=300)
    plt.close()

    print(f'Saved {out_name}')


def main():
    for rho, path in DENSITY_FILES.items():
        make_rse_plot(rho, path)


if __name__ == '__main__':
    main()
