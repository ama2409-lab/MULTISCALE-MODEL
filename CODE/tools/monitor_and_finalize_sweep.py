import argparse
import csv
import math
import os
import shutil
import time
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_results(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def aggregate(rows):
    # group by L_m (as float)
    groups = defaultdict(list)
    for r in rows:
        try:
            L = float(r.get('L_m', r.get('L', r.get('L_m', '0'))))
        except Exception:
            continue
        try:
            val_dir = float(r.get('dir_D_eff', r.get('dir_D_eff', 'nan')))
        except Exception:
            val_dir = float('nan')
        try:
            val_neu = float(r.get('neu_D_eff', r.get('neu_D_eff', 'nan')))
        except Exception:
            val_neu = float('nan')
        groups[L].append((val_dir, val_neu))

    xs = []
    dir_mean = []
    dir_sem = []
    neu_mean = []
    neu_sem = []
    for L in sorted(groups.keys()):
        vals = groups[L]
        dirs = [v[0] for v in vals if math.isfinite(v[0])]
        neus = [v[1] for v in vals if math.isfinite(v[1])]
        n = max(1, max(len(dirs), len(neus)))
        if dirs:
            mean_d = sum(dirs) / len(dirs)
            std_d = math.sqrt(sum((x - mean_d) ** 2 for x in dirs) / len(dirs))
            sem_d = std_d / math.sqrt(len(dirs))
        else:
            mean_d = float('nan')
            sem_d = float('nan')
        if neus:
            mean_n = sum(neus) / len(neus)
            std_n = math.sqrt(sum((x - mean_n) ** 2 for x in neus) / len(neus))
            sem_n = std_n / math.sqrt(len(neus))
        else:
            mean_n = float('nan')
            sem_n = float('nan')
        xs.append(L)
        dir_mean.append(mean_d)
        dir_sem.append(sem_d)
        neu_mean.append(mean_n)
        neu_sem.append(sem_n)

    return xs, dir_mean, dir_sem, neu_mean, neu_sem


def plot_and_save(xs, dir_mean, dir_sem, neu_mean, neu_sem, out_path):
    plt.figure(figsize=(6, 4))
    # plot Dirichlet
    plt.errorbar(xs, dir_mean, yerr=dir_sem, fmt='o-', label='Dirichlet')
    plt.errorbar(xs, neu_mean, yerr=neu_sem, fmt='s--', label='Neumann')
    plt.xscale('log')
    plt.xlabel('Domain size L (m)')
    plt.ylabel('Effective diffusivity (D_eff)')
    plt.legend()
    plt.grid(True, which='both', ls=':', lw=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def finalize(csv_path, expected_count, results_folder_base=None):
    rows = read_results(csv_path)
    if len(rows) < expected_count:
        print(f"Not enough rows yet: {len(rows)} / {expected_count}")
        return False

    xs, dir_mean, dir_sem, neu_mean, neu_sem = aggregate(rows)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if results_folder_base is None:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        results_folder_base = f"FINAL SET/{base}_final_{timestamp}"

    os.makedirs(results_folder_base, exist_ok=True)

    plot_path = os.path.join(results_folder_base, f'sweep_rev_const_density_plot_final_{timestamp}.png')
    print(f"Saving final plot to {plot_path}")
    plot_and_save(xs, dir_mean, dir_sem, neu_mean, neu_sem, plot_path)

    # move CSV into folder (create a copy to keep original in place? we move it)
    dst_csv = os.path.join(results_folder_base, os.path.basename(csv_path))
    print(f"Moving CSV {csv_path} -> {dst_csv}")
    try:
        shutil.move(csv_path, dst_csv)
    except Exception as e:
        print(f"Failed to move CSV: {e}")
        return False

    print("Finalization complete.")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to results CSV to monitor')
    parser.add_argument('--expected', type=int, default=192, help='Expected number of result rows')
    parser.add_argument('--interval', type=int, default=30, help='Polling interval in seconds')
    args = parser.parse_args()

    csv_path = args.csv
    expected = args.expected
    interval = args.interval

    print(f"Monitoring {csv_path} for {expected} rows (poll every {interval}s)")
    while True:
        if os.path.exists(csv_path):
            try:
                rows = read_results(csv_path)
                count = len(rows)
            except Exception as e:
                print(f"Error reading CSV: {e}")
                count = 0
        else:
            count = 0
        print(f"Found {count} rows (expect {expected})")
        if count >= expected:
            ok = finalize(csv_path, expected)
            if ok:
                break
            else:
                print("Finalization failed, will retry in a bit.")
        time.sleep(interval)


if __name__ == '__main__':
    main()
