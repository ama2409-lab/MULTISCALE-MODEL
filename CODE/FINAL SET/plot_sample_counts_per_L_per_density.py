import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Map density labels to their sweep files
DENSITY_FILES = {
    1e9: Path('const_density_1e09') / 'sweep_const_density_1e09_full.csv',
    1e12: Path('const_density_1e12') / 'sweep_const_density_1e12_full.csv',
    1e13: Path('const_density_1e13') / 'sweep_const_density_1e13_full.csv',
    1e14: Path('const_density_1e14') / 'sweep_const_density_1e14_full.csv',
}


def make_plot_for_density(rho, csv_path):
    if not csv_path.is_file():
        print(f'Warning: file not found for density {rho:.0e}: {csv_path}')
        return

    df = pd.read_csv(csv_path)

    if 'L_m' not in df.columns:
        print(f"No 'L_m' column in {csv_path}; skipping.")
        return

    # Optionally ignore failed runs
    if 'error' in df.columns:
        df = df[df['error'].isna() | (df['error'] == '')]

    counts = df.groupby('L_m').size().reset_index(name='count')
    counts = counts.sort_values('L_m')

    L = counts['L_m'].to_numpy()
    n = counts['count'].to_numpy()

    plt.figure(figsize=(6, 4))
    plt.plot(L, n, marker='o')
    plt.xscale('log')
    plt.xlabel('Domain size L [m]')
    plt.ylabel('Number of samples (seeds)')
    plt.title(f'Sample count per L for density {rho:.0e} pores/m³')
    plt.grid(True, which='both', linestyle=':', alpha=0.4)
    plt.tight_layout()

    out_name = f'sample_counts_L_density_{int(rho):d}.png'
    plt.savefig(out_name, dpi=300)
    plt.close()

    print(f'Saved {out_name}')


def main():
    for rho, path in DENSITY_FILES.items():
        make_plot_for_density(rho, path)


if __name__ == '__main__':
    main()
