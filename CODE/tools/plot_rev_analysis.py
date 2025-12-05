import argparse
import csv
import math
import os
from collections import defaultdict
from datetime import datetime

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


def compute_stats(rows):
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
        groups[L]['dir'].append(d)
        groups[L]['neu'].append(n)

    Ls = sorted(groups.keys())
    stats = []
    for L in Ls:
        dirs = [v for v in groups[L]['dir'] if math.isfinite(v)]
        neus = [v for v in groups[L]['neu'] if math.isfinite(v)]

        def summarize(arr):
            if not arr:
                return {
                    'mean': math.nan,
                    'std': math.nan,
                    'sem': math.nan,
                    'n': 0,
                    'sum': 0.0,
                    'sum_sq': 0.0,
                }
            n = len(arr)
            s1 = float(sum(arr))
            s2 = float(sum(a * a for a in arr))
            mean = s1 / n
            if n > 1:
                var = max(0.0, (s2 - (s1 * s1) / n) / (n - 1))
                std = math.sqrt(var)
                sem = std / math.sqrt(n)
            else:
                std = math.nan
                sem = math.nan
            return {
                'mean': mean,
                'std': std,
                'sem': sem,
                'n': n,
                'sum': s1,
                'sum_sq': s2,
            }

        d_stats = summarize(dirs)
        n_stats = summarize(neus)
        stats.append({
            'L': L,
            'd_mean': d_stats['mean'],
            'd_std': d_stats['std'],
            'd_sem': d_stats['sem'],
            'd_n': d_stats['n'],
            'd_sum': d_stats['sum'],
            'd_sum_sq': d_stats['sum_sq'],
            'n_mean': n_stats['mean'],
            'n_std': n_stats['std'],
            'n_sem': n_stats['sem'],
            'n_n': n_stats['n'],
            'n_sum': n_stats['sum'],
            'n_sum_sq': n_stats['sum_sq'],
        })

    for idx, item in enumerate(stats):
        d_mean = item['d_mean']
        n_mean = item['n_mean']
        item['d_cov'] = abs(item['d_std'] / d_mean) if item['d_std'] and d_mean else math.nan
        item['n_cov'] = abs(item['n_std'] / n_mean) if item['n_std'] and n_mean else math.nan
        if idx == 0:
            item['d_rel_prev'] = math.nan
            item['n_rel_prev'] = math.nan
        else:
            prev = stats[idx - 1]
            prev_d_mean = prev['d_mean']
            prev_n_mean = prev['n_mean']
            if prev_d_mean and d_mean and math.isfinite(prev_d_mean) and math.isfinite(d_mean):
                item['d_rel_prev'] = abs(d_mean - prev_d_mean) / abs(prev_d_mean)
            else:
                item['d_rel_prev'] = math.nan
            if prev_n_mean and n_mean and math.isfinite(prev_n_mean) and math.isfinite(n_mean):
                item['n_rel_prev'] = abs(n_mean - prev_n_mean) / abs(prev_n_mean)
            else:
                item['n_rel_prev'] = math.nan

    return stats


def combine_running(sum_val, sum_sq, count):
    if count <= 0:
        return math.nan, math.nan
    mean = sum_val / count
    if count > 1:
        var = max(0.0, (sum_sq - (sum_val * sum_val) / count) / (count - 1))
        sem = math.sqrt(var) / math.sqrt(count)
        return mean, sem
    return mean, math.nan


def compute_running(stats):
    running = {
        'L': [],
        'd_mean': [],
        'd_sem': [],
        'd_n': [],
        'n_mean': [],
        'n_sem': [],
        'n_n': [],
    }
    cum_d_sum = 0.0
    cum_d_sum_sq = 0.0
    cum_d_n = 0
    cum_n_sum = 0.0
    cum_n_sum_sq = 0.0
    cum_n_n = 0
    for item in stats:
        cum_d_sum += item['d_sum']
        cum_d_sum_sq += item['d_sum_sq']
        cum_d_n += item['d_n']
        cum_n_sum += item['n_sum']
        cum_n_sum_sq += item['n_sum_sq']
        cum_n_n += item['n_n']

        d_mean, d_sem = combine_running(cum_d_sum, cum_d_sum_sq, cum_d_n)
        n_mean, n_sem = combine_running(cum_n_sum, cum_n_sum_sq, cum_n_n)

        running['L'].append(item['L'])
        running['d_mean'].append(d_mean)
        running['d_sem'].append(d_sem)
        running['d_n'].append(cum_d_n)
        running['n_mean'].append(n_mean)
        running['n_sem'].append(n_sem)
        running['n_n'].append(cum_n_n)

    return running


def find_rev(stats, rel_thresh=0.05, cov_thresh=0.1, min_count=50):
    if len(stats) < 2:
        return None
    for item in stats[1:]:
        d_ok = (
            item['d_n'] >= min_count and
            math.isfinite(item['d_rel_prev']) and item['d_rel_prev'] <= rel_thresh and
            math.isfinite(item['d_cov']) and item['d_cov'] <= cov_thresh
        )
        n_ok = (
            item['n_n'] >= min_count and
            math.isfinite(item['n_rel_prev']) and item['n_rel_prev'] <= rel_thresh and
            math.isfinite(item['n_cov']) and item['n_cov'] <= cov_thresh
        )
        if d_ok and n_ok:
            return {
                'L': item['L'],
                'd_rel_prev': item['d_rel_prev'],
                'd_cov': item['d_cov'],
                'd_count': item['d_n'],
                'n_rel_prev': item['n_rel_prev'],
                'n_cov': item['n_cov'],
                'n_count': item['n_n'],
            }
    return None


def safe_array(vals):
    return np.array([np.nan if (v is None or not math.isfinite(v)) else v for v in vals], dtype=float)


def plot_stats(stats, running, out_prefix, rel_thresh, cov_thresh, rev_candidate=None):
    Ls = safe_array([s['L'] for s in stats])
    d_means = safe_array([s['d_mean'] for s in stats])
    d_sems = safe_array([s['d_sem'] for s in stats])
    n_means = safe_array([s['n_mean'] for s in stats])
    n_sems = safe_array([s['n_sem'] for s in stats])
    counts = [max(s['d_n'], s['n_n']) for s in stats]

    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    ci_scale = 1.96
    d_ci = ci_scale * d_sems
    n_ci = ci_scale * n_sems

    ax0 = axes[0]
    ax0.fill_between(Ls, d_means - d_ci, d_means + d_ci, color='#1f77b4', alpha=0.2, label='Dirichlet 95% CI')
    ax0.fill_between(Ls, n_means - n_ci, n_means + n_ci, color='#ff7f0e', alpha=0.2, label='Neumann 95% CI')
    ax0.plot(Ls, d_means, marker='o', color='#1f77b4', linewidth=1.5, label='Dirichlet mean')
    ax0.plot(Ls, n_means, marker='s', linestyle='--', color='#ff7f0e', linewidth=1.5, label='Neumann mean')
    ax0.set_ylabel('D_eff (m$^2$/s)')
    ax0.set_xscale('log')
    ax0.grid(True, which='both', linestyle=':', linewidth=0.7)
    ax0.legend(loc='best')

    ax0_twin = ax0.twinx()
    ax0_twin.bar(Ls, counts, width=0.15 * Ls, color='0.8', alpha=0.5, label='Samples per L')
    ax0_twin.set_ylabel('Samples')
    ax0_twin.set_ylim(0, max(counts + [10]) * 1.2)
    ax0_twin.set_yticks([20, 40, 60, 80, 100, 120])
    ax0_twin.grid(False)

    handles, labels = ax0.get_legend_handles_labels()
    twin_handles, twin_labels = ax0_twin.get_legend_handles_labels()
    ax0.legend(handles + twin_handles, labels + twin_labels, loc='upper left')

    ax1 = axes[1]
    d_rel_prev = safe_array([s['d_rel_prev'] for s in stats])
    n_rel_prev = safe_array([s['n_rel_prev'] for s in stats])
    ax1.plot(Ls, d_rel_prev, marker='o', color='#1f77b4', linewidth=1.5, label='Dirichlet |Δ prev| / prev')
    ax1.plot(Ls, n_rel_prev, marker='s', linestyle='--', color='#ff7f0e', linewidth=1.5, label='Neumann |Δ prev| / prev')
    ax1.axhline(rel_thresh, color='k', linestyle=':', linewidth=1.0, label=f'Rel threshold {rel_thresh:.2f}')
    ax1.set_yscale('log')
    ax1.set_ylabel('Relative change')
    ax1.grid(True, which='both', linestyle=':', linewidth=0.7)
    ax1.legend(loc='best')

    ax2 = axes[2]
    run_Ls = safe_array(running['L'])
    run_d_mean = safe_array(running['d_mean'])
    run_d_sem = safe_array(running['d_sem'])
    run_n_mean = safe_array(running['n_mean'])
    run_n_sem = safe_array(running['n_sem'])
    run_d_ci = ci_scale * run_d_sem
    run_n_ci = ci_scale * run_n_sem
    ax2.fill_between(run_Ls, run_d_mean - run_d_ci, run_d_mean + run_d_ci, color='#1f77b4', alpha=0.2)
    ax2.fill_between(run_Ls, run_n_mean - run_n_ci, run_n_mean + run_n_ci, color='#ff7f0e', alpha=0.2)
    ax2.plot(run_Ls, run_d_mean, marker='o', color='#1f77b4', linewidth=1.5, label='Dirichlet running mean')
    ax2.plot(run_Ls, run_n_mean, marker='s', linestyle='--', color='#ff7f0e', linewidth=1.5, label='Neumann running mean')
    ax2.set_xlabel('L (m)')
    ax2.set_ylabel('Running D_eff (m$^2$/s)')
    ax2.set_xscale('log')
    ax2.grid(True, which='both', linestyle=':', linewidth=0.7)
    ax2.legend(loc='best')

    if rev_candidate:
        L_rev = rev_candidate['L']
        for ax in axes:
            ax.axvline(L_rev, color='green', linestyle='--', linewidth=1.2)
        ax0.annotate(f"REV candidate\nL={L_rev:.3g} m", xy=(L_rev, np.nanmax(d_means+n_means)),
                     xytext=(5, 15), textcoords='offset points', color='green', fontsize=9,
                     arrowprops=dict(arrowstyle='->', color='green'))
        ax1.annotate('REV candidate', xy=(L_rev, rel_thresh), xytext=(5, -15), textcoords='offset points',
                     color='green', fontsize=9)
        ax2.annotate('REV candidate', xy=(L_rev, np.nanmax(run_d_mean+run_n_mean)),
                     xytext=(5, 10), textcoords='offset points', color='green', fontsize=9)

    axes[-1].set_xlim(Ls.min(), Ls.max())
    axes[-1].set_xscale('log')
    axes[-1].set_xlabel('Domain size L (m)')

    plt.tight_layout()
    out_path = out_prefix + '_dashboard.png'
    plt.savefig(out_path, dpi=250)
    plt.close(fig)

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--rel-thresh', type=float, default=0.05,
                        help='Relative difference threshold to previous L (default 0.05)')
    parser.add_argument('--cov-thresh', type=float, default=0.1,
                        help='Coefficient of variation threshold (default 0.1)')
    parser.add_argument('--min-count', type=int, default=50,
                        help='Minimum number of samples required before declaring REV (default 50)')
    args = parser.parse_args()

    rows = read_rows(args.csv)
    stats = compute_stats(rows)
    if not stats:
        print('No stats found.')
        return

    running = compute_running(stats)

    out_dir = os.path.dirname(args.csv)
    base = os.path.splitext(os.path.basename(args.csv))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_prefix = os.path.join(out_dir, f'{base}_rev_analysis_{timestamp}')
    cand = find_rev(stats, rel_thresh=args.rel_thresh, cov_thresh=args.cov_thresh, min_count=args.min_count)
    dashboard = plot_stats(stats, running, out_prefix, args.rel_thresh, args.cov_thresh, cand)

    if cand is None:
        print('No REV found with thresholds: rel<=%.3f, COV<=%.3f, min_count=%d' % (
            args.rel_thresh, args.cov_thresh, args.min_count))
    else:
        print('REV candidate at L = %.4g m' % cand['L'])
        print('  Dirichlet: rel_prev=%.4g, COV=%.4g (n=%d)' % (
            cand['d_rel_prev'], cand['d_cov'], cand['d_count']))
        print('  Neumann:   rel_prev=%.4g, COV=%.4g (n=%d)' % (
            cand['n_rel_prev'], cand['n_cov'], cand['n_count']))

    print('Dashboard saved to:', dashboard)


if __name__ == '__main__':
    main()
