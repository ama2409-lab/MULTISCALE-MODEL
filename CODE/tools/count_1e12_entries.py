#!/usr/bin/env python3
import csv
from collections import Counter, defaultdict
from pathlib import Path

# Paths to 1e12 CSVs (discovered earlier)
csv_paths = [
    Path(r"CODE/FINAL SET/const_density_1e12/sweep_const_density_1e12_20251123_203902.csv"),
    Path(r"CODE/FINAL SET/const_density_1e12/sweep_const_density_1e12_20251123_203956.csv"),
    Path(r"CODE/FINAL SET/const_density_1e12/sweep_const_density_1e12_20251123_212514.csv"),
]

counter = Counter()
np_sums = defaultdict(float)
nt_sums = defaultdict(float)
np_counts = Counter()
rows_read = 0
missing_files = []
for p in csv_paths:
    if not p.exists():
        missing_files.append(str(p))
        continue
    with p.open('r', newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows_read += 1
            L = r.get('L_m') or r.get('L')
            if L is None:
                continue
            counter[L] += 1
            if r.get('Np'):
                try:
                    np_sums[L] += float(r['Np'])
                    np_counts[L] += 1
                except Exception:
                    pass
            if r.get('Nt'):
                try:
                    nt_sums[L] += float(r['Nt'])
                except Exception:
                    pass

print(f"Checked {len(csv_paths)} paths, rows read: {rows_read}")
if missing_files:
    print("Missing files:")
    for f in missing_files:
        print(" -", f)

# Sort L values numerically where possible
def tryfloat(x):
    try:
        return float(x)
    except Exception:
        return float('inf')

items = sorted(counter.items(), key=lambda kv: tryfloat(kv[0]))

TARGET = 100
stats = {}
print('\nCounts per L_m (sorted):')
print('L_m,count,missing_to_100,avg_Np,avg_Nt')
for L, c in items:
    miss = max(0, TARGET - c)
    avg_np = np_sums[L] / np_counts[L] if np_counts[L] else ''
    avg_nt = nt_sums[L] / np_counts[L] if np_counts[L] else ''
    stats[L] = {
        'count': c,
        'missing': miss,
        'avg_np': avg_np if avg_np != '' else None,
        'avg_nt': avg_nt if avg_nt != '' else None,
    }
    print(f"{L},{c},{miss},{avg_np},{avg_nt}")

print('\nSummary:')
print(f"Unique L values: {len(items)}")
print(f"Total rows: {rows_read}")

# Print which L values have >= TARGET
ok = [L for L,c in items if c >= TARGET]
need = [L for L,c in items if c < TARGET]
print(f"L with >= {TARGET} realizations: {len(ok)}")
print(f"L needing more runs: {len(need)}")
if need:
    print('List needing more runs:')
    for L in need:
        print(' -', L)

# Also print per-L approx Np,Nt by reading first matching row from files
print('\nDone.')
pilot_path = Path(r"CODE/FINAL SET/pilot_const_density_1e12.csv")
if pilot_path.exists():
    try:
        import numpy as np

        pilot_np = []
        pilot_t = []
        with pilot_path.open('r', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    pilot_np.append(float(row['Np']))
                    pilot_t.append(float(row['t_total']))
                except Exception:
                    continue

        if len(pilot_np) >= 2:
            A = np.vstack([pilot_np, np.ones(len(pilot_np))]).T
            slope, intercept = np.linalg.lstsq(A, pilot_t, rcond=None)[0]
            print('\nRuntime model (from pilot_const_density_1e12.csv):')
            print(f"  t_estimate ≈ {slope:.6e} * Np + {intercept:.3f}  [seconds]")
            total_missing_seconds = 0.0
            print('\nEstimated runtime per L (seconds):')
            for L, data in stats.items():
                avg_np = data['avg_np']
                if avg_np is None:
                    continue
                est = slope * avg_np + intercept
                est = max(est, 0.0)
                total_missing_seconds += data['missing'] * est
                print(f"  L={L}: est_per_run={est:.2f}, missing_runs={data['missing']}, est_total={est * data['missing']:.1f}")
            hours = total_missing_seconds / 3600.0
            print(f"\nTotal estimated time to reach {TARGET} per L: {total_missing_seconds:.1f} s (~{hours:.2f} h)")
        else:
            print('\nPilot file found but insufficient rows to fit runtime model.')
    except Exception as exc:
        print('\nCould not compute runtime model:', exc)
else:
    print('\nPilot runtime file not found; skipping runtime estimates.')
