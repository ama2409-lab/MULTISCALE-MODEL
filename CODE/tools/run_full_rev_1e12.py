#!/usr/bin/env python3
"""Generate >=100 realizations per domain size for a constant pore density."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import sys

ROOT = Path(__file__).resolve().parents[1]
FINAL_SET = ROOT / "FINAL SET"
if str(FINAL_SET) not in sys.path:
    sys.path.insert(0, str(FINAL_SET))

from sweep_rev_sizes import process_L_seeds  # noqa: E402

L_VALUES = [
    1.0e-08,
    1.58489319246111e-08,
    2.51188643150958e-08,
    3.98107170553497e-08,
    1.0e-07,
    1.58489319246111e-07,
    2.51188643150958e-07,
    3.98107170553497e-07,
    1.0e-06,
    1.58489319246111e-06,
    2.51188643150958e-06,
    3.98107170553497e-06,
    5.0e-06,
    1.0e-05,
    1.58489319246111e-05,
    2.0e-05,
    2.51188643150958e-05,
    3.98107170553497e-05,
    1.0e-04,
    1.58489319246111e-04,
    2.51188643150958e-04,
    3.98107170553497e-04,
    1.0e-03,
    1.58489319246111e-03,
    2.51188643150958e-03,
]

HEADER = ['timestamp', 'seed', 'L_m', 'Np', 'Nt', 'td_fixed_count', 'tl_fixed_count',
          'cond_n_throats', 'cond_n_positive', 'dir_rate', 'dir_D_eff',
          'neu_rate_imposed', 'neu_rate_calc', 'neu_D_eff', 'porosity',
          'tau_dir', 'tau_neu', 'density', 'error']


def format_density_tag(density: float) -> str:
    if density <= 0:
        raise ValueError('Density must be positive.')
    return f"{density:.0e}".replace('+', '')


def discover_existing_csvs(density: float, out_path: Path, extra: str | None) -> list[Path]:
    tag = format_density_tag(density)
    default_dir = FINAL_SET / f"const_density_{tag}"
    paths = []
    if extra:
        for item in extra.split(','):
            item = item.strip()
            if item:
                paths.append(Path(item))
    else:
        pattern = f"sweep_const_density_{tag}_*.csv"
        if default_dir.exists():
            paths.extend(sorted(default_dir.glob(pattern)))
    # Avoid loading the live output file unless --resume handles it explicitly later
    paths = [p for p in paths if p.resolve() != out_path.resolve()]
    return paths


def load_existing(paths):
    rows = []
    per_L = defaultdict(lambda: {'rows': [], 'seeds': set()})
    for path in paths:
        if not path.exists():
            continue
        with path.open('r', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                L_raw = row.get('L_m')
                if not L_raw:
                    continue
                seed = row.get('seed')
                try:
                    L = float(L_raw)
                except Exception:
                    continue
                key = round(L, 12)
                per_L[key]['rows'].append(row)
                if seed is not None and seed != '':
                    per_L[key]['seeds'].add(int(float(seed)))
                rows.append(row)
    return rows, per_L


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', type=int, default=100, help='Target realizations per L (default: 100)')
    ap.add_argument('--seed-start', type=int, default=1000, help='Starting seed for new runs (default: 1000)')
    ap.add_argument('--chunk', type=int, default=1, help='Number of seeds to run per process_L_seeds call (default: 1)')
    ap.add_argument('--density', type=float, default=1e12, help='Pore density (default: 1e12)')
    ap.add_argument('--rate', type=float, default=1e-12, help='Total rate for Neumann BC (default: 1e-12)')
    ap.add_argument('--thickness-fraction', type=float, default=0.05, help='Boundary slice thickness fraction (default: 0.05)')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--existing', help='Comma-separated list of CSVs that already contain results to seed counts/resume logic. Defaults to all sweep_const_density_{density}_*.csv in the matching FINAL SET directory.')
    ap.add_argument('--resume', action='store_true', help='If set, re-use rows already present in --out to avoid duplication.')
    ap.add_argument('--dry-run', action='store_true', help='Report how many runs are needed without executing them.')
    ap.add_argument('--only', help='Comma-separated subset of L values to process (e.g. 1e-03,2.51188643150958e-03).')
    return ap.parse_args()


def ensure_output(out_path: Path, existing_rows):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)


def main():
    args = parse_args()

    density_tag = format_density_tag(args.density)
    default_dir = FINAL_SET / f"const_density_{density_tag}"

    if args.out is None:
        args.out = default_dir / f"sweep_const_density_{density_tag}_full.csv"
    else:
        args.out = Path(args.out)

    existing_paths = discover_existing_csvs(args.density, args.out, args.existing)

    active_L = L_VALUES
    if args.only:
        try:
            subset = [float(x.strip()) for x in args.only.split(',') if x.strip()]
            mapping = {L: L for L in L_VALUES}
            active_L = []
            for val in subset:
                # match float to canonical value
                key = min(mapping.keys(), key=lambda k: abs(k - val))
                candidate = mapping[key]
                if candidate not in active_L:
                    active_L.append(candidate)
        except Exception:
            print('Could not parse --only list; falling back to full list.')
            active_L = L_VALUES

    existing_rows, per_L = load_existing(existing_paths)

    if args.resume and args.out.exists():
        extra_rows, extra_per_L = load_existing([args.out])
        existing_rows.extend(extra_rows)
        for L, info in extra_per_L.items():
            per_L[L]['rows'].extend(info['rows'])
            per_L[L]['seeds'].update(info['seeds'])

    if not args.dry_run:
        ensure_output(args.out, existing_rows)

    total_new = 0
    total_time = 0.0
    next_seed = args.seed_start

    for L in active_L:
        key = round(L, 12)
        existing_count = len(per_L.get(key, {}).get('rows', []))
        missing = args.target - existing_count
        if missing <= 0:
            print(f"L={L:.3e} already has {existing_count} runs (target {args.target}); skipping.")
            continue

        used_seeds = set(per_L.get(key, {}).get('seeds', set()))
        new_seeds = []
        while len(new_seeds) < missing:
            if next_seed not in used_seeds:
                new_seeds.append(next_seed)
                used_seeds.add(next_seed)
            next_seed += 1

        print(f"L={L:.3e}: existing={existing_count}, new_runs={len(new_seeds)}")

        if args.dry_run:
            total_new += len(new_seeds)
            continue

        for i in range(0, len(new_seeds), args.chunk):
            chunk = new_seeds[i:i + args.chunk]
            print(f"  -> seeds {chunk[0]}..{chunk[-1]} ({len(chunk)} runs)")
            t0 = perf_counter()
            process_L_seeds(
                L,
                chunk,
                density=args.density,
                out=str(args.out),
                plot=str(args.out.with_suffix('.png')),
                rate=args.rate,
                thickness_fraction=args.thickness_fraction,
                save_datasets=False,
            )
            dt = perf_counter() - t0
            total_time += dt
            total_new += len(chunk)
            print(f"     completed in {dt/len(chunk):.2f} s/run ({dt:.1f} s for chunk)")

    if args.dry_run:
        print(f"\nDry-run total new runs needed: {total_new}")
    else:
        print(f"\nCompleted {total_new} new runs in {total_time/3600:.2f} h ({total_time:.1f} s).")
        print(f"Results appended to {args.out}")


if __name__ == '__main__':
    main()
