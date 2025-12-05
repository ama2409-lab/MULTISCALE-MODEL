import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# List all sweep files to include (relative to this script directory)
SWEEP_FILES = [
    Path('const_density_1e09') / 'sweep_const_density_1e09_full.csv',
    Path('const_density_1e12') / 'sweep_const_density_1e12_full.csv',
    Path('const_density_1e13') / 'sweep_const_density_1e13_full.csv',
    Path('const_density_1e14') / 'sweep_const_density_1e14_full.csv',
]

OUT_FIG = 'dirichlet_neumann_ratio_vs_L_all_sweeps.png'


def load_all():
    frames = []
    for p in SWEEP_FILES:
        if not p.is_file():
            print(f'Warning: file not found, skipping: {p}')
            continue
        df = pd.read_csv(p)
        df['source_file'] = str(p)
        frames.append(df)
    if not frames:
        raise RuntimeError('No sweep files were found; check paths in SWEEP_FILES.')
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_all()

    # Dirichlet / Neumann column names used in the REV sweeps
    possible_dir_cols = ['dir_D_eff', 'D_eff_dirichlet', 'Deff_dir', 'Deff_D']
    possible_neu_cols = ['neu_D_eff', 'D_eff_neumann', 'Deff_neu', 'Deff_N']

    dir_col = next((c for c in possible_dir_cols if c in df.columns), None)
    neu_col = next((c for c in possible_neu_cols if c in df.columns), None)

    if dir_col is None or neu_col is None:
        print('Could not find Dirichlet/Neumann D_eff columns in combined CSVs.')
        print('Available columns:', df.columns.tolist())
        return

    if 'L_m' not in df.columns:
        print("No 'L_m' column found; cannot plot versus domain size.")
        print('Available columns:', df.columns.tolist())
        return

    # Filter out error rows if present
    if 'error' in df.columns:
        df = df[df['error'].isna() | (df['error'] == '')]

    d_dir = df[dir_col].to_numpy()
    d_neu = df[neu_col].to_numpy()
    L = df['L_m'].to_numpy()

    mask = np.isfinite(d_dir) & np.isfinite(d_neu) & np.isfinite(L)
    d_dir = d_dir[mask]
    d_neu = d_neu[mask]
    L = L[mask]

    if d_dir.size == 0:
        print('No valid data after filtering.')
        return

    ratio = d_neu / np.maximum(d_dir, 1e-30)

    # Slight jitter in L to separate overlapping points
    jitter = (np.random.rand(len(L)) - 0.5) * 0.03 * L
    L_jitter = L + jitter

    plt.figure(figsize=(7, 4))
    # Plot ±10% band as horizontal region over full L range of data
    L_min, L_max = L.min(), L.max()
    plt.fill_between([L_min, L_max], 0.9, 1.1, color='gray', alpha=0.2, label='±10% band')
    plt.axhline(1.0, color='k', linestyle='--', label='Perfect agreement')
    plt.scatter(L_jitter, ratio, s=15, alpha=0.7)
    plt.xscale('log')
    plt.ylim(0.0, 2.0)
    plt.xlabel('Domain size L [m]')
    plt.ylabel(r'$D_\mathrm{eff}^{(Neumann)} / D_\mathrm{eff}^{(Dirichlet)}$')
    plt.title('Dirichlet–Neumann agreement across all sweeps (L, density, seeds)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    plt.close()

    print(f'Saved ratio-vs-L figure (all sweeps) to {OUT_FIG}')
    print(f'Mean ratio: {ratio.mean():.3f}, std: {ratio.std():.3f}')
    frac_in_08_10 = np.mean((ratio >= 0.8) & (ratio <= 1.0))
    print(f'Fraction of points with 0.8 <= ratio <= 1.0: {frac_in_08_10:.1%}')


if __name__ == '__main__':
    main()
