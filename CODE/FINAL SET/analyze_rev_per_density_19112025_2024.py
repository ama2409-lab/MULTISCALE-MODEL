"""Generate REV convergence plots (Dirichlet and Neumann) per density.

Reads the *_full.csv files in the const_density_* folders and produces
D_eff vs L plots with error bars for each density, similar to
`const_density_1e13/sweep_rev_const_density_plot_final_20251119_225450.png`.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path('.')
DENSITY_FOLDERS = {
    1e9: BASE / 'const_density_1e09',
    1e12: BASE / 'const_density_1e12',
    1e13: BASE / 'const_density_1e13',
    1e14: BASE / 'const_density_1e14',
}

CSV_NAMES = {
    1e9: 'sweep_const_density_1e09_full.csv',
    1e12: 'sweep_const_density_1e12_full.csv',
    1e13: 'sweep_const_density_1e13_full.csv',
    1e14: 'sweep_const_density_1e14_full.csv',
}

for rho, folder in DENSITY_FOLDERS.items():
    csv_path = folder / CSV_NAMES[rho]
    if not csv_path.exists():
        print(f"[WARN] Missing CSV for density {rho:.0e}: {csv_path}")
        continue

    print(f"\n=== Density {rho:.0e} ===")
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Basic cleaning
    if 'error' in df.columns:
        df = df[(df['error'].isna()) | (df['error'] == '')].copy()
    df = df.dropna(subset=['dir_D_eff', 'neu_D_eff'])

    # Group by L and seed to get stats
    grouped = df.groupby('L_m').agg({
        'dir_D_eff': ['mean', 'std', 'count'],
        'neu_D_eff': ['mean', 'std', 'count'],
    })

    Ls = grouped.index.values
    dir_mean = grouped['dir_D_eff']['mean'].values
    dir_std = grouped['dir_D_eff']['std'].values
    dir_se = dir_std / np.sqrt(grouped['dir_D_eff']['count'].values)
    neu_mean = grouped['neu_D_eff']['mean'].values
    neu_std = grouped['neu_D_eff']['std'].values
    neu_se = neu_std / np.sqrt(grouped['neu_D_eff']['count'].values)

    # =============================
    # Plot 1: D_eff vs L (REV curve)
    # =============================
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(Ls * 1e6, dir_mean, yerr=dir_se, marker='o', linestyle='-', linewidth=2,
                label='Dirichlet', capsize=4, markersize=6, color='#2E86AB')
    ax.errorbar(Ls * 1e6, neu_mean, yerr=neu_se, marker='s', linestyle='--', linewidth=2,
                label='Neumann', capsize=4, markersize=6, color='#A23B72')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Domain size L (μm)', fontsize=12)
    ax.set_ylabel('Effective diffusivity D_eff (m²/s)', fontsize=12)
    ax.set_title(f'REV convergence at density {rho:.0e} (pores/m³)', fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend()

    out_path = folder / f'rev_convergence_{int(rho):d}.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved REV convergence plot: {out_path}")

    # =============================
    # Plot 2: COV vs L (std/mean)
    # =============================
    cov_dir = np.where(dir_mean != 0, np.abs(dir_std / dir_mean), np.nan)
    cov_neu = np.where(neu_mean != 0, np.abs(neu_std / neu_mean), np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(Ls, cov_dir, marker='o', linestyle='-', color='#2E86AB', label='Dirichlet COV')
    ax.plot(Ls, cov_neu, marker='s', linestyle='--', color='#F39C12', label='Neumann COV')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('L (m)', fontsize=12)
    ax.set_ylabel('COV (std/mean)', fontsize=12)
    ax.set_title(f'COV of D_eff vs L at density {rho:.0e}', fontsize=14)
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    ax.legend()

    cov_path = folder / f'cov_rev_{int(rho):d}.png'
    plt.tight_layout()
    plt.savefig(cov_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved COV plot: {cov_path}")

print("\nAll per-density REV and COV plots generated.")
