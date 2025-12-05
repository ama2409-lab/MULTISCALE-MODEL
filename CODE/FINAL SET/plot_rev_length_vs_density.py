import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Densities and corresponding sweep files (const-density data)
DENSITY_FILES = {
    1e9: Path('const_density_1e09') / 'sweep_const_density_1e09_full.csv',
    1e12: Path('const_density_1e12') / 'sweep_const_density_1e12_full.csv',
    1e13: Path('const_density_1e13') / 'sweep_const_density_1e13_full.csv',
    1e14: Path('const_density_1e14') / 'sweep_const_density_1e14_full.csv',
}

OUT_FIG = 'rev_length_vs_density.png'
OUT_FIG_REG = 'rev_length_vs_density_with_regression.png'


def find_rev_L_for_density(csv_path, skip=False,
                           change_thresh=0.05,
                           bc_diff_thresh=0.10,
                           min_L_fraction=0.3):
    """Return approximate REV length for one density.

    Criteria:
      - Use Dirichlet and Neumann mean D_eff vs L.
      - Walk L from small to large, and mark L_k as REV if:
          * relative change |D_k - D_{k-1}| / D_k < change_thresh, and
          * relative BC difference |D_dir_k - D_neu_k| / D_dir_k < bc_diff_thresh.
      - If never satisfied, return largest L as lower bound and flag as such.
    """
    if not csv_path.is_file():
        print(f'Skipping missing file: {csv_path}')
        return None, None

    df = pd.read_csv(csv_path)
    if 'L_m' not in df.columns:
        print(f"No 'L_m' column in {csv_path}; skipping.")
        return None, None

    # Filter out failed runs
    if 'error' in df.columns:
        df = df[df['error'].isna() | (df['error'] == '')]

    dir_cols = ['dir_D_eff', 'D_eff_dirichlet', 'Deff_dir', 'Deff_D']
    neu_cols = ['neu_D_eff', 'D_eff_neumann', 'Deff_neu', 'Deff_N']

    dir_col = next((c for c in dir_cols if c in df.columns), None)
    neu_col = next((c for c in neu_cols if c in df.columns), None)

    if dir_col is None or neu_col is None:
        print(f'Could not find D_eff columns in {csv_path}. Columns: {df.columns.tolist()}')
        return None, None

    grouped = df.groupby('L_m').agg(
        dir_mean=(dir_col, 'mean'),
        neu_mean=(neu_col, 'mean'),
    ).reset_index().sort_values('L_m')

    L = grouped['L_m'].to_numpy()
    d_dir = grouped['dir_mean'].to_numpy()
    d_neu = grouped['neu_mean'].to_numpy()

    if skip:
        # For 1e9: we might not want to include; return None
        return None, None

    # Walk from second point onwards and look for first L_k that satisfies both criteria.
    # To avoid declaring an REV at tiny L (where curves are still evolving a lot),
    # only start checking once L has reached a given fraction of the max L.
    L_max = L[-1]
    L_min_check = L_max * min_L_fraction
    rev_L = None
    is_lower_bound = True
    for k in range(1, len(L)):
        if L[k] < L_min_check:
            continue
        if d_dir[k] == 0 or d_dir[k-1] == 0:
            continue
        rel_change = abs(d_dir[k] - d_dir[k-1]) / abs(d_dir[k])
        rel_bc = abs(d_neu[k] - d_dir[k]) / abs(d_dir[k])
        if (rel_change < change_thresh) and (rel_bc < bc_diff_thresh):
            rev_L = L[k]
            is_lower_bound = False
            break

    if rev_L is None:
        # No clear REV found; use largest L as lower bound
        rev_L = L[-1]
        is_lower_bound = True

    return rev_L, is_lower_bound


def main():
    densities = []
    rev_Ls = []
    lower_bounds = []

    for rho, path in DENSITY_FILES.items():
        # Skip 1e9 as requested
        skip = (rho == 1e9)
        L_rev, is_lb = find_rev_L_for_density(path, skip=skip)
        if L_rev is None:
            continue
        densities.append(rho)
        rev_Ls.append(L_rev)
        lower_bounds.append(is_lb)
        print(f'rho={rho:.0e}: L_REV ~ {L_rev:.3e} m (lower_bound={is_lb})')

    if not densities:
        print('No REV points computed.')
        return

    densities = np.array(densities)
    rev_Ls = np.array(rev_Ls)
    lower_bounds = np.array(lower_bounds, dtype=bool)

    # --- Original scatter plot (no regression) ---
    plt.figure(figsize=(6, 4))
    plt.loglog(densities, rev_Ls, 'o', color='tab:blue')
    plt.xlabel('Pore density 03c [pores/m$^3$]')
    plt.ylabel('REV length $L_{REV}$ [m]')
    plt.title('Approximate REV length vs pore density (skip 1e9)')
    plt.grid(True, which='both', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    plt.close()
    print(f'Saved {OUT_FIG}')

    # --- Log-log regression using the available densities (currently 1e12, 1e13, 1e14) ---
    log_rho = np.log10(densities)
    log_L = np.log10(rev_Ls)
    slope, intercept = np.polyfit(log_rho, log_L, deg=1)

    # Predict L_REV at rho = 1e9 using the regression
    rho_pred = 1e9
    log_rho_pred = np.log10(rho_pred)
    log_L_pred = slope * log_rho_pred + intercept
    L_pred = 10 ** log_L_pred
    print(f'Predicted L_REV at rho=1e9 from regression: {L_pred:.3e} m')

    # Build smooth regression line over a range of densities
    rho_min = min(rho_pred, densities.min())
    rho_max = densities.max()
    rho_grid = np.logspace(np.log10(rho_min), np.log10(rho_max), 100)
    log_rho_grid = np.log10(rho_grid)
    log_L_grid = slope * log_rho_grid + intercept
    L_grid = 10 ** log_L_grid

    # Plot data, regression line, and predicted 1e9 point
    plt.figure(figsize=(6, 4))
    plt.loglog(densities, rev_Ls, 'o', color='tab:blue', label='Estimated $L_{REV}$')
    plt.loglog(rho_grid, L_grid, '-', color='tab:orange', label='Log-log regression')
    plt.loglog(rho_pred, L_pred, 's', color='tab:red', label=r'Predicted $L_{REV}$ at $10^9$')

    plt.xlabel('Pore density 03c [pores/m$^3$]')
    plt.ylabel('REV length $L_{REV}$ [m]')
    plt.title('REV length vs pore density with regression (includes 1e9 prediction)')
    plt.grid(True, which='both', linestyle=':', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG_REG, dpi=300)
    plt.close()

    print(f'Saved {OUT_FIG_REG}')


if __name__ == '__main__':
    main()
