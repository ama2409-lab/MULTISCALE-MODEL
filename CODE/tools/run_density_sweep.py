#!/usr/bin/env python3
"""Run sweeps at an arbitrary pore density by batching seeds per-domain-size.

This driver imports the in-process `process_L_seeds` function from
`FINAL SET/sweep_rev_sizes.py` when possible (fast, avoids repeated heavy
imports). If that import fails it falls back to calling the script as a
subprocess for each (L,seed) pair.

Example (PowerShell):
    python tools/run_density_sweep.py --density 1e12 --sizes 1e-05,1e-04 --seeds 100-103

Outputs a timestamped CSV into `FINAL SET/const_density_<density>/` by default
and runs `tools/rev_diagnostics.py` at the end to produce diagnostics.
"""
from __future__ import annotations
import argparse
import os
import sys
import subprocess
from datetime import datetime
from typing import List


def parse_sizes(s: str) -> List[float]:
    return [float(x) for x in s.split(',') if x.strip()]


def parse_seeds(s: str) -> List[int]:
    s = s.strip()
    if '-' in s:
        a, b = s.split('-', 1)
        return list(range(int(a), int(b) + 1))
    if ',' in s:
        return [int(x) for x in s.split(',') if x.strip()]
    return [int(s)]


def density_str(d: float) -> str:
    # Make a compact label like 1e13
    return f"{d:.0e}".replace('+', '')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--density', type=float, required=True)
    p.add_argument('--sizes', required=True, help='Comma list of L (m), e.g. 1e-5,1e-4')
    p.add_argument('--seeds', default='100-107', help='Range or list, e.g. 100-107 or 100,101')
    p.add_argument('--outdir', default=None, help='Directory to save CSV and diagnostics')
    p.add_argument('--rate', type=float, default=1e-12, help='Neumann total rate')
    p.add_argument('--thickness-fraction', type=float, default=0.05)
    p.add_argument('--save-datasets', action='store_true', help='Save per-sample .npz datasets using ml_dataset_exporter')
    p.add_argument('--dataset-dir', default=None, help='Base directory for saved datasets (overrides default DATASETS/...)')
    p.add_argument('--expected', type=int, default=None, help='Expected number of rows for monitor finalization')
    args = p.parse_args()

    sizes = parse_sizes(args.sizes)
    seeds = parse_seeds(args.seeds)

    # Prepare directories
    cwd = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(cwd, '..'))
    final_set_dir = os.path.join(repo_root, 'FINAL SET')
    os.makedirs(final_set_dir, exist_ok=True)
    dstr = density_str(args.density)
    outdir = args.outdir or os.path.join(final_set_dir, f'const_density_{dstr}')
    os.makedirs(outdir, exist_ok=True)

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_csv = os.path.join(outdir, f'sweep_const_density_{dstr}_{timestamp}.csv')

    # Prefer importing the in-process batch helper to avoid repeated heavy imports
    proc_impl = None
    try:
        sys.path.insert(0, final_set_dir)
        import sweep_rev_sizes as srs

        if hasattr(srs, 'process_L_seeds'):
            proc_impl = srs.process_L_seeds
            print('Using in-process `process_L_seeds` from FINAL SET/sweep_rev_sizes.py')
    except Exception as e:
        print('Could not import in-process sweep helper (will fall back to subprocess):', e)

    # Run the work: for each L, run all seeds in one batch
    for L in sizes:
        seed_list = list(seeds)
        print(f'Processing L={L:.3g} m with seeds={seed_list} -> writing to {out_csv}')
        if proc_impl is not None:
            try:
                proc_impl(L, seed_list, density=args.density, out=out_csv,
                          plot=os.path.join(outdir, f'sweep_rev_plot_{dstr}_{timestamp}.png'),
                          rate=args.rate, thickness_fraction=args.thickness_fraction,
                          save_datasets=args.save_datasets, dataset_dir=args.dataset_dir)
            except Exception as e:
                print(f'In-process call failed for L={L}: {e} -- falling back to subprocess for this L')
                proc_impl = None

        if proc_impl is None:
            # fallback: call the FINAL SET script as a subprocess for each seed
            script_path = os.path.join(final_set_dir, 'sweep_rev_sizes.py')
            for seed in seed_list:
                cmd = [sys.executable, script_path,
                       '--sizes', str(L), '--density', str(args.density), '--seed', str(seed),
                       '--out', out_csv, '--plot', os.path.join(outdir, f'sweep_rev_plot_{dstr}_{timestamp}.png'),
                       '--rate', str(args.rate), '--thickness-fraction', str(args.thickness_fraction)]
                print('Running subprocess:', ' '.join(cmd))
                subprocess.check_call(cmd)

    # After the runs, invoke diagnostics to summarize
    try:
        diag_script = os.path.join(repo_root, 'tools', 'rev_diagnostics.py')
        print('Running diagnostics...')
        subprocess.check_call([sys.executable, diag_script, '--csv', out_csv, '--outdir', outdir, '--bootstrap', '2000'])
    except Exception as e:
        print('Diagnostics script failed:', e)

    # Optionally finalize via monitor if the user provided expected count
    if args.expected:
        try:
            mon = os.path.join(repo_root, 'tools', 'monitor_and_finalize_sweep.py')
            print(f'Finalizing with monitor (expected={args.expected})')
            subprocess.check_call([sys.executable, mon, '--csv', out_csv, '--expected', str(args.expected)])
        except Exception as e:
            print('Monitor/finalize step failed:', e)

    print('Sweep driver finished. Results CSV:', out_csv)


if __name__ == '__main__':
    main()
