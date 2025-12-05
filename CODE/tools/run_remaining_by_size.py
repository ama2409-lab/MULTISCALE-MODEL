import os, sys, subprocess
import pandas as pd
import numpy as np

folder = r"FINAL SET"
files = [f for f in os.listdir(folder) if f.startswith('sweep_rev_const_density_results_') and f.endswith('.csv')]
if not files:
    print('No existing sweep CSV found. Abort.')
    sys.exit(1)
files = sorted(files)
csv_name = files[-1]
csv_path = os.path.join(folder, csv_name)
print('Using CSV:', csv_path)

df = pd.read_csv(csv_path)
df['L_m'] = pd.to_numeric(df['L_m'], errors='coerce')
completed = set((float(r['L_m']), int(r['seed'])) for _, r in df.dropna(subset=['L_m','seed']).iterrows())

# desired sizes and seeds
sizes = []
for d in range(-8, -2):
    vals = np.logspace(d, d+1, num=5, endpoint=False)
    sizes.extend(vals[:4].tolist())
sizes = sorted(set(sizes))
seed_base = 100
seeds = [seed_base + i for i in range(8)]

all_pairs = [(float(L), s) for L in sizes for s in seeds]
remaining = [p for p in all_pairs if p not in completed]
print(f'Total desired runs: {len(all_pairs)}, completed: {len(completed)}, remaining: {len(remaining)}')

from collections import defaultdict
byL = defaultdict(list)
for L, s in remaining:
    byL[L].append(s)

for L in sorted(byL.keys()):
    seeds_list = sorted(byL[L])
    for seed in seeds_list:
        cmd = [sys.executable, os.path.join('FINAL SET','sweep_rev_sizes.py'), '--sizes', str(L), '--density', str(1e13), '--seed', str(seed), '--out', csv_path, '--plot', os.path.join('FINAL SET', csv_name.replace('.csv', '.png')), '--rate', str(1e-12), '--thickness-fraction', str(0.05)]
        print(f'Running L={L:.3e} seed={seed}')
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print('Run failed for L=', L, ' seed=', seed, ' exit=', r.returncode)

print('All batches submitted/completed')
