import subprocess, sys, os
import numpy as np
from datetime import datetime
import pandas as pd

# Define sizes and seeds used in original study
sizes = []
for d in range(-8, -2):
    vals = np.logspace(d, d+1, num=5, endpoint=False)
    sizes.extend(vals[:4].tolist())
sizes = sorted(set(sizes))
seed_base = 100
seeds_per_size = 8
seeds = [seed_base + i for i in range(seeds_per_size)]

density = 1e13
rate = 1e-12
thickness_fraction = 0.05

# Detect latest CSV matching pattern
folder = r"FINAL SET"
files = [f for f in os.listdir(folder) if f.startswith('sweep_rev_const_density_results_') and f.endswith('.csv')]
if not files:
    print('No existing sweep CSV found. Run the full sweep runner instead.')
    sys.exit(1)
files = sorted(files)
csv_name = files[-1]
csv_path = os.path.join(folder, csv_name)
print('Resuming sweep using CSV:', csv_path)

# Read completed runs
try:
    df = pd.read_csv(csv_path)
    df['L_m'] = pd.to_numeric(df['L_m'], errors='coerce')
    completed = set((float(r['L_m']), int(r['seed'])) for _, r in df.dropna(subset=['L_m','seed']).iterrows())
except Exception as e:
    print('Could not read existing CSV:', e)
    completed = set()

# Build list of all desired combos
all_pairs = [(float(L), s) for L in sizes for s in seeds]
remaining = [p for p in all_pairs if p not in completed]
print(f'Total runs: {len(all_pairs)}, Completed: {len(completed)}, Remaining: {len(remaining)}')

if not remaining:
    print('No remaining runs. Exiting.')
    sys.exit(0)

# Derive plot filename from csv base
base_ts = csv_name.replace('sweep_rev_const_density_results_','').replace('.csv','')
plot = os.path.join(folder, f"sweep_rev_const_density_plot_{base_ts}.png")

# Run remaining runs
for L, seed in remaining:
    size_str = f"{L}"
    print(f"Running L={L:.3e}, seed={seed}")
    cmd = [sys.executable, r"FINAL SET\sweep_rev_sizes.py", "--sizes", size_str, "--density", str(density), "--seed", str(seed), "--out", csv_path, "--plot", plot, "--rate", str(rate), "--thickness-fraction", str(thickness_fraction)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"Run failed: L={L}, seed={seed} (exit {r.returncode})")

print('Resume sweep complete. Final CSV:', csv_path)
print('Plot file:', plot)
