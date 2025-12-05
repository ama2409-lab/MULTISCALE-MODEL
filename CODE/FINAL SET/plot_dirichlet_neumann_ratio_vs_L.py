import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CSV_FILE = 'sweep_rev_results.csv'
OUT_FIG = 'dirichlet_neumann_ratio_vs_L.png'


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

    if 'L_m' not in df.columns:
        print("No 'L_m' column found; cannot plot versus domain size.")
        print('Available columns:', df.columns.tolist())
        return

    # Filter out error rows if present
    if 'error' in df.columns:
        df = df[df['error'].isna() | (df['error'] == '')]

    x_dir = df[dir_col].to_numpy()
    x_neu = df[neu_col].to_numpy()
    L = df['L_m'].to_numpy()

    mask = np.isfinite(x_dir) & np.isfinite(x_neu) & np.isfinite(L)
    x_dir = x_dir[mask]
    x_neu = x_neu[mask]
    L = L[mask]

    if x_dir.size == 0:
        print('No valid data after filtering.')
        return

    ratio = x_neu / np.maximum(x_dir, 1e-30)

    # Slight jitter in L to separate overlapping points
    jitter = (np.random.rand(len(L)) - 0.5) * 0.03 * L
    L_jitter = L + jitter

    plt.figure(figsize=(7, 4))
    # Plot ±10% band as horizontal region over fixed L range
    L_min, L_max = 1e-8, 1e-3
    plt.fill_between([L_min, L_max], 0.9, 1.1, color='gray', alpha=0.2, label='±10% band')
    plt.axhline(1.0, color='k', linestyle='--', label='Perfect agreement')
    plt.scatter(L_jitter, ratio, s=20, alpha=0.7)
    plt.xscale('log')
    plt.ylim(0.8, 1.2)
    plt.xlabel('Domain size L [m]')
    plt.ylabel(r'$D_\mathrm{eff}^{(Neumann)} / D_\mathrm{eff}^{(Dirichlet)}$')
    plt.title('Dirichlet–Neumann agreement across domain sizes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    plt.close()

    print(f'Saved ratio-vs-L figure to {OUT_FIG}')
    print(f'Mean ratio: {ratio.mean():.3f}, std: {ratio.std():.3f}')


if __name__ == '__main__':
    main()
