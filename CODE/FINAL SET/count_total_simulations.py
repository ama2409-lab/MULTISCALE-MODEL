import pandas as pd
from pathlib import Path

FILES = [
    Path('sweep_rev_results.csv'),
    Path('sweep_rev_const_density_results.csv'),
    Path('sweep_rev_const_density_results_19112025_1649.csv'),
    Path('sweep_rev_results_backup_20251119_160922.csv'),
    Path('const_density_1e09') / 'sweep_const_density_1e09_full.csv',
    Path('const_density_1e12') / 'sweep_const_density_1e12_full.csv',
    Path('const_density_1e13') / 'sweep_const_density_1e13_full.csv',
    Path('const_density_1e14') / 'sweep_const_density_1e14_full.csv',
]


def main():
    total = 0
    for p in FILES:
        if not p.is_file():
            print(f'Skipping missing file: {p}')
            continue
        df = pd.read_csv(p)
        n = len(df)
        print(f'{p}: {n} rows')
        total += n
    print('-' * 40)
    print(f'Total simulations (all listed CSVs): {total}')


if __name__ == '__main__':
    main()
