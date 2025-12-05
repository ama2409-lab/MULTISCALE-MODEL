import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CSV_FILE = 'sweep_rev_results.csv'
SCATTER_FIG = 'dirichlet_neumann_largeL_scatter.png'
RATIO_FIG = 'dirichlet_neumann_ratio_largeL.png'


def main():
    df = pd.read_csv(CSV_FILE)

    # Column names (adapt if yours differ)
    possible_dir_cols = ['dir_D_eff', 'D_eff_dirichlet', 'Deff_dir', 'Deff_D']
    possible_neu_cols = ['neu_D_eff', 'D_eff_neumann', 'Deff_neu', 'Deff_N']

    dir_col = next((c for c in possible_dir_cols if c in df.columns), None)
    neu_col = next((c for c in possible_neu_cols if c in df.columns), None)

    if dir_col is None or neu_col is None:
        print('Could not find Dirichlet/Neumann D_eff columns in CSV.')
        print('Available columns:', df.columns.tolist())
        return

    # Filter out error rows if present
    if 'error' in df.columns:
        df = df[df['error'].isna() | (df['error'] == '')]

    if 'L_m' not in df.columns:
        print("No 'L_m' column found; cannot select largest domains.")
        return

    # Select only the largest domain size (strongest REV regime)
    max_L = df['L_m'].max()
    df_large = df[df['L_m'] == max_L].copy()

    if df_large.empty:
        print('No rows for largest L.')
        return

    x = df_large[dir_col].to_numpy()
    y = df_large[neu_col].to_numpy()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        print('No valid data at largest L after filtering.')
        return

    # --- Scatter at largest L ---
    dmin = min(x.min(), y.min())
    dmax = max(x.max(), y.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=25, alpha=0.7)
    plt.plot([dmin, dmax], [dmin, dmax], 'k--', label='1:1')
    plt.xlabel(r'$D_\mathrm{eff}^{(Dirichlet)}$ [m$^2$/s]')
    plt.ylabel(r'$D_\mathrm{eff}^{(Neumann)}$ [m$^2$/s]')
    plt.title(f'Dirichlet vs Neumann at largest L = {max_L:.2e} m')
    plt.legend()
    plt.tight_layout()
    plt.savefig(SCATTER_FIG, dpi=300)
    plt.close()

    # --- Ratio plot at largest L ---
    ratio = y / np.maximum(x, 1e-30)
    idx = np.arange(len(ratio))

    plt.figure(figsize=(7, 4))
    plt.axhline(1.0, color='k', linestyle='--', label='Perfect agreement')
    plt.fill_between(idx, 0.9, 1.1, color='gray', alpha=0.2, label='±10% band')
    plt.plot(idx, ratio, 'o', ms=5)
    plt.xlabel('Realization index (seeds at largest L)')
    plt.ylabel(r'$D_\mathrm{eff}^{(Neumann)} / D_\mathrm{eff}^{(Dirichlet)}$')
    plt.title('Dirichlet–Neumann agreement at largest L')
    plt.ylim(0.8, 1.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RATIO_FIG, dpi=300)
    plt.close()

    print(f'Saved scatter to {SCATTER_FIG}')
    print(f'Saved ratio plot to {RATIO_FIG}')
    print(f'Mean ratio: {ratio.mean():.3f}, std: {ratio.std():.3f}')


if __name__ == '__main__':
    main()
