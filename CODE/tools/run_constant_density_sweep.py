density = 1e13
import subprocess, sys
import numpy as np
from datetime import datetime
import os

# Generate sizes: for each decade from 1e-8 to 1e-2 (decades -8..-2) pick 4 values per decade
sizes = []
for d in range(-8, -2):
    vals = np.logspace(d, d+1, num=5, endpoint=False)
    sizes.extend(vals[:4].tolist())
# unique sorted
sizes = sorted(set(sizes))
# seeds: 8 different seeds per size
seed_base = 100
seeds_per_size = 8
seeds = [seed_base + i for i in range(seeds_per_size)]

density = 1e13
rate = 1e-12
thickness_fraction = 0.05

# Create a Windows-safe timestamp (no colons) in the format DDMMYYYY_HHMM
ts = datetime.now().strftime('%d%m%Y_%H%M')
out_csv = fr"FINAL SET/sweep_rev_const_density_results_{ts}.csv"
plot = fr"FINAL SET/sweep_rev_const_density_plot_{ts}.png"

os.makedirs(r"FINAL SET", exist_ok=True)

print(f"Running sweep: {len(sizes)} sizes, {len(seeds)} seeds each -> total runs = {len(sizes)*len(seeds)}")
print(f"Output CSV: {out_csv}")
print(f"Output plot: {plot}")
print("Sizes (m):")
for s in sizes:
    print(f"  {s:.3e}")

# Run runs
for L in sizes:
    size_str = f"{L}"
    for si, seed in enumerate(seeds):
        print(f"Running L={L:.3e}, seed={seed}")
        cmd = [sys.executable, r"FINAL SET\sweep_rev_sizes.py", "--sizes", size_str, "--density", str(density), "--seed", str(seed), "--out", out_csv, "--plot", plot, "--rate", str(rate), "--thickness-fraction", str(thickness_fraction)]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"Run failed: L={L}, seed={seed} (exit {r.returncode})")
print("Sweep complete")
print(f"Final outputs saved to {out_csv} and {plot}")
