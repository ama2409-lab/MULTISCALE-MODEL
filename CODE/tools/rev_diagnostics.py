import argparse
import csv
import math
import os
import shutil
from collections import defaultdict
from datetime import datetime
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_rows(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def per_L_stats(rows, out_csv=None):
    groups = defaultdict(lambda: {'dir': [], 'neu': []})
    for r in rows:
        try:
            L = float(r.get('L_m', r.get('L', None)))
        except Exception:
            continue
        try:
            d = float(r.get('dir_D_eff', 'nan'))
        except Exception:
            d = float('nan')
        try:
            n = float(r.get('neu_D_eff', 'nan'))
        except Exception:
            n = float('nan')
        if math.isfinite(d):
            groups[L]['dir'].append(d)
        if math.isfinite(n):
            groups[L]['neu'].append(n)

    Ls = sorted(groups.keys())
    stats = []
    for L in Ls:
        dirs = groups[L]['dir']
        neus = groups[L]['neu']
        def summarize(arr):
            if not arr:
                return (math.nan, math.nan, 0)
            arr = np.array(arr)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=0))
            sem = float(std / math.sqrt(len(arr)))
            return (mean, std, sem, len(arr))
        d_mean, d_std, d_sem, d_n = summarize(dirs)
        n_mean, n_std, n_sem, n_n = summarize(neus)
        d_cov = d_std / d_mean if d_mean and d_mean != 0 else math.nan
        n_cov = n_std / n_mean if n_mean and n_mean != 0 else math.nan
        stats.append({'L': L, 'd_mean': d_mean, 'd_std': d_std, 'd_sem': d_sem, 'd_n': d_n, 'd_cov': d_cov,
                      'n_mean': n_mean, 'n_std': n_std, 'n_sem': n_sem, 'n_n': n_n, 'n_cov': n_cov})

    if out_csv:
        fieldnames = ['L', 'd_mean', 'd_std', 'd_sem', 'd_n', 'd_cov', 'n_mean', 'n_std', 'n_sem', 'n_n', 'n_cov']
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in stats:
                writer.writerow(s)

    return stats, groups


def box_violin_plots(groups, out_prefix):
    Ls = sorted(groups.keys())
    labels = [f"{L:.3g}" for L in Ls]

    # Dirichlet boxplot
    dir_data = [groups[L]['dir'] for L in Ls]
    plt.figure(figsize=(10,4))
    plt.boxplot(dir_data, labels=labels, showfliers=False)
    plt.yscale('log')
    plt.xlabel('L (m)')
    plt.ylabel('Dirichlet D_eff')
    plt.title('Dirichlet D_eff distribution across seeds')
    plt.tight_layout()
    p1 = out_prefix + '_dir_box.png'
    plt.savefig(p1)
    plt.close()

    # Neumann boxplot
    neu_data = [groups[L]['neu'] for L in Ls]
    plt.figure(figsize=(10,4))
    plt.boxplot(neu_data, labels=labels, showfliers=False)
    plt.yscale('log')
    plt.xlabel('L (m)')
    plt.ylabel('Neumann D_eff')
    plt.title('Neumann D_eff distribution across seeds')
    plt.tight_layout()
    p2 = out_prefix + '_neu_box.png'
    plt.savefig(p2)
    plt.close()

    # Violin combined (Dir and Neu side by side)
    plt.figure(figsize=(12,5))
    pos = np.arange(len(Ls))
    # matplotlib's violinplot is awkward with empty groups; handle gracefully
    dir_plot_data = [g if g else [0] for g in dir_data]
    neu_plot_data = [g if g else [0] for g in neu_data]
    plt.violinplot(dir_plot_data, positions=pos - 0.2, widths=0.35, showmeans=False)
    plt.violinplot(neu_plot_data, positions=pos + 0.2, widths=0.35, showmeans=False)
    plt.xticks(pos, labels)
    plt.yscale('log')
    plt.xlabel('L (m)')
    plt.ylabel('D_eff')
    plt.title('Violin plots of D_eff (Dir=left, Neu=right)')
    plt.tight_layout()
    p3 = out_prefix + '_violin.png'
    plt.savefig(p3)
    plt.close()

    return p1, p2, p3


def bootstrap_mean_ci(arr, nboot=1000, alpha=0.05):
    arr = np.array(arr)
    if arr.size == 0:
        return (math.nan, math.nan)
    means = []
    for _ in range(nboot):
        sample = np.random.choice(arr, size=arr.size, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, 100 * (alpha/2))
    upper = np.percentile(means, 100 * (1 - alpha/2))
    return lower, upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--outdir', default=None)
    args = parser.parse_args()

    rows = read_rows(args.csv)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.dirname(args.csv)
    outdir = args.outdir or base_dir
    os.makedirs(outdir, exist_ok=True)

    stats_csv = os.path.join(outdir, f'rev_stats_{timestamp}.csv')
    stats, groups = per_L_stats(rows, out_csv=stats_csv)

    out_prefix = os.path.join(outdir, f'rev_diag_{timestamp}')
    p1, p2, p3 = box_violin_plots(groups, out_prefix)

    # bootstrap for candidate Ls: smallest L and largest L and reported candidate
    Ls = sorted(groups.keys())
    candidate_L = None
    if Ls:
        candidate_L = Ls[len(Ls)//2]  # just pick middle if none specified
    boot_results = {}
    for L in [Ls[0] if Ls else None, candidate_L, Ls[-1] if Ls else None]:
        if L is None:
            continue
        dir_arr = groups[L]['dir']
        neu_arr = groups[L]['neu']
        dir_ci = bootstrap_mean_ci(dir_arr, nboot=args.bootstrap) if dir_arr else (math.nan, math.nan)
        neu_ci = bootstrap_mean_ci(neu_arr, nboot=args.bootstrap) if neu_arr else (math.nan, math.nan)
        boot_results[L] = {'dir_ci': dir_ci, 'neu_ci': neu_ci, 'dir_n': len(dir_arr), 'neu_n': len(neu_arr)}

    # write a small report
    report = os.path.join(outdir, f'rev_report_{timestamp}.txt')
    with open(report, 'w') as f:
        f.write('REV Diagnostics Report\n')
        f.write(f'CSV: {args.csv}\n')
        f.write(f'STATS CSV: {stats_csv}\n')
        f.write('\nPer-L summary:\n')
        for s in stats:
            f.write(f"L={s['L']}: d_mean={s['d_mean']}, d_std={s['d_std']}, d_n={s['d_n']}, d_cov={s['d_cov']}\n")
        f.write('\nBootstrap CIs (95%):\n')
        for L, br in boot_results.items():
            f.write(f"L={L}: dir_CI={br['dir_ci']} (n={br['dir_n']}), neu_CI={br['neu_ci']} (n={br['neu_n']})\n")

    print('Diagnostics complete. Outputs:')
    print(' -', stats_csv)
    print(' -', p1)
    print(' -', p2)
    print(' -', p3)
    print(' -', report)

    # Create an organized folder and copy key outputs into it
    org_dir = os.path.join(os.path.dirname(outdir), f'organized_{timestamp}')
    os.makedirs(org_dir, exist_ok=True)
    for path in [stats_csv, p1, p2, p3, report]:
        try:
            shutil.copy(path, org_dir)
        except Exception:
            pass
    print('Copied key outputs to', org_dir)


if __name__ == '__main__':
    main()
