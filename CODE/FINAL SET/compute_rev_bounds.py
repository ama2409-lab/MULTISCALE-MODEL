#!/usr/bin/env python3
"""Run DIRICHLET and NEUMANN scripts and compute REV upper/lower bounds.

This script executes the two provided scripts as subprocesses, parses their
stdout for Effective diffusivity, porosity, and tortuosity, and writes a CSV
summary. It intentionally does NOT modify the original scripts.

Usage (PowerShell):
    python compute_rev_bounds.py
    python compute_rev_bounds.py --dir script_path --neu script_path --out results.csv

"""
import argparse
import csv
import datetime
import os
import re
import subprocess
import sys

DEFAULT_DIR = r"FINAL SET\FICKIAN_VORONOI_ON_NETWORK_DIRICHLET.py"
DEFAULT_NEU = r"FINAL SET\FICKIAN_VORONOI_ON_NETWORK_NEUMANN.py"

PATTERNS = {
    'D_eff': re.compile(r'Effective diffusivity:\s*([0-9Ee+\-\.]+)'),
    'porosity': re.compile(r'Porosity:\s*([0-9Ee+\-\.]+)'),
    'tau': re.compile(r'Tortuosity(?: \(Dirichlet\)| \(Neumann\))?:\s*([0-9Ee+\-\.]+)'),
    'rate': re.compile(r'Molar flow rate:\s*([0-9Ee+\-\.]+)')
}


def run_script(path):
    """Run a python script and capture stdout/stderr. Returns a dict with text and returncode."""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return {'ok': False, 'error': f'File not found: {abs_path}', 'stdout': ''}
    cmd = [sys.executable, abs_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {'ok': True, 'rc': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
    except Exception as e:
        return {'ok': False, 'error': str(e), 'stdout': ''}


def parse_metrics(text):
    out = {'D_eff': None, 'porosity': None, 'tau': None, 'rate': None}
    for k, pat in PATTERNS.items():
        m = pat.search(text)
        if m:
            try:
                out[k] = float(m.group(1))
            except Exception:
                out[k] = None
    return out


def append_csv(out_path, row, header=None):
    exists = os.path.exists(out_path)
    with open(out_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default=DEFAULT_DIR, help='Path to Dirichlet script')
    p.add_argument('--neu', default=DEFAULT_NEU, help='Path to Neumann script')
    p.add_argument('--out', default='rev_bounds_results.csv', help='Output CSV file')
    p.add_argument('--skip-dir', action='store_true', help="Don't run Dirichlet script")
    p.add_argument('--skip-neu', action='store_true', help="Don't run Neumann script")
    args = p.parse_args()

    timestamp = datetime.datetime.utcnow().isoformat()

    results = {
        'timestamp': timestamp,
        'dir_path': args.dir,
        'neu_path': args.neu,
        'dir_rc': None,
        'neu_rc': None,
        'dir_stdout_snippet': '',
        'neu_stdout_snippet': '',
        'dir_D_eff': None,
        'neu_D_eff': None,
        'dir_porosity': None,
        'neu_porosity': None,
        'dir_tau': None,
        'neu_tau': None,
        'dir_rate': None,
        'neu_rate': None,
    }

    if not args.skip_dir:
        print(f'Running Dirichlet script: {args.dir}')
        r = run_script(args.dir)
        if not r.get('ok'):
            print('Dirichlet run failed:', r.get('error'))
        else:
            results['dir_rc'] = r.get('rc')
            results['dir_stdout_snippet'] = '\n'.join(r.get('stdout').splitlines()[-40:])
            parsed = parse_metrics(r.get('stdout'))
            results['dir_D_eff'] = parsed.get('D_eff')
            results['dir_porosity'] = parsed.get('porosity')
            results['dir_tau'] = parsed.get('tau')
            results['dir_rate'] = parsed.get('rate')

    if not args.skip_neu:
        print(f'Running Neumann script: {args.neu}')
        r = run_script(args.neu)
        if not r.get('ok'):
            print('Neumann run failed:', r.get('error'))
        else:
            results['neu_rc'] = r.get('rc')
            results['neu_stdout_snippet'] = '\n'.join(r.get('stdout').splitlines()[-40:])
            parsed = parse_metrics(r.get('stdout'))
            results['neu_D_eff'] = parsed.get('D_eff')
            results['neu_porosity'] = parsed.get('porosity')
            results['neu_tau'] = parsed.get('tau')
            results['neu_rate'] = parsed.get('rate')

    # Compute simple REV bounds (Dirichlet upper, Neumann lower if present)
    dir_val = results['dir_D_eff']
    neu_val = results['neu_D_eff']
    if dir_val is not None and neu_val is not None:
        results['REV_upper'] = dir_val
        results['REV_lower'] = neu_val
        results['REV_ratio'] = dir_val / neu_val if neu_val != 0 else None
    else:
        results['REV_upper'] = dir_val
        results['REV_lower'] = neu_val
        results['REV_ratio'] = None

    # CSV header and row
    header = [
        'timestamp', 'dir_path', 'neu_path', 'dir_rc', 'neu_rc',
        'dir_D_eff', 'neu_D_eff', 'REV_upper', 'REV_lower', 'REV_ratio',
        'dir_porosity', 'neu_porosity', 'dir_tau', 'neu_tau', 'dir_rate', 'neu_rate',
        'dir_stdout_snippet', 'neu_stdout_snippet'
    ]
    row = {k: results.get(k, '') for k in header}

    append_csv(args.out, row, header=header)
    print('\nResults:')
    print(f"  Dirichlet D_eff = {results['dir_D_eff']}")
    print(f"  Neumann  D_eff = {results['neu_D_eff']}")
    print(f"  REV bounds: upper={results.get('REV_upper')}, lower={results.get('REV_lower')}")
    print(f"  Saved results to {args.out}")


if __name__ == '__main__':
    main()
