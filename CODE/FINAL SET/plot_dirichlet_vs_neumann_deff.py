import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Adjust this filename if your main sweep CSV has a different name
CSV_FILE = 'sweep_rev_results.csv'
OUT_FIG = 'dirichlet_vs_neumann_deff_scatter.png'


def main():
    df = pd.read_csv(CSV_FILE)

    # Try to infer Dirichlet/Neumann columns
    # Update these names if your CSV uses slightly different ones
    possible_dir_cols = ['dir_D_eff', 'D_eff_dirichlet', 'Deff_dir', 'Deff_D']
    possible_neu_cols = ['neu_D_eff', 'D_eff_neumann', 'Deff_neu', 'Deff_N']

    dir_col = next((c for c in possible_dir_cols if c in df.columns), None)
    neu_col = next((c for c in possible_neu_cols if c in df.columns), None)

    if dir_col is None or neu_col is None:
        print('Could not find Dirichlet/Neumann D_eff columns in CSV.')
        print('Available columns:', df.columns.tolist())
        return

    # Drop rows with NaNs or errors if there is an error column
    if 'error' in df.columns:
        df = df[df['error'].isna() | (df['error'] == '')]

    x = df[dir_col].values
    y = df[neu_col].values

    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        print('No valid D_eff data after filtering.')
        return

    # Compute 1:1 limits
    dmin = min(x.min(), y.min())
    dmax = max(x.max(), y.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=15, alpha=0.6)
    plt.plot([dmin, dmax], [dmin, dmax], 'k--', label='1:1')
    plt.xlabel(r'$D_\mathrm{eff}^{(Dirichlet)}$ [m$^2$/s]')
    plt.ylabel(r'$D_\mathrm{eff}^{(Neumann)}$ [m$^2$/s]')
    plt.title('Dirichlet vs Neumann Effective Diffusivity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    plt.close()

    # Simple relative-difference summary
    rel_diff = np.abs(y - x) / np.maximum(x, 1e-30)
    print(f'Saved scatter figure to {OUT_FIG}')
    print(f'Mean relative difference: {rel_diff.mean():.2%}')
    print(f'Median relative difference: {np.median(rel_diff):.2%}')


if __name__ == '__main__':
    main()
