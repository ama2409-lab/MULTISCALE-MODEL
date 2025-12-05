#!/usr/bin/env python3
"""Run pilot realizations at density 1e12 to estimate per-sample runtime."""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from time import perf_counter
from datetime import datetime

import numpy as np
import openpnm as op

# Ensure imports work relative to repository root
ROOT = Path(__file__).resolve().parents[1]
FINAL_SET = ROOT / "FINAL SET"
if str(FINAL_SET) not in sys.path:
    sys.path.insert(0, str(FINAL_SET))

from sweep_rev_sizes import (  # noqa: E402
    conservative_geometry_repairs,
    compute_robust_conductance,
    find_inlet_outlet_pores,
    cluster_restrict,
    run_dirichlet,
    run_neumann,
)
from NETWORK_GENERATION_FINAL import build_voronoi_network  # noqa: E402


def run_single(L: float, seed: int, density: float = 1e12,
               rate: float = 1e-12, thickness_fraction: float = 0.05,
               pore_size_dist=None, throat_size_dist=None,
               correlate_pore_throat: bool = False,
               beta_params=(5.0, 2.0)):
    """Run one realization and capture timing breakdown."""
    pore_size_dist = pore_size_dist or {'name': 'lognorm', 's': 0.25, 'scale': 20e-6}
    throat_size_dist = throat_size_dist or {'name': 'weibull_min', 'c': 2.5, 'scale': 8e-6}

    timings = {}
    start_total = perf_counter()

    num_points = int(max(10, round(density * (L ** 3))))

    t0 = perf_counter()
    pn = build_voronoi_network(
        domain_size=(L, L, L),
        pore_density=density,
        points=num_points,
        pore_size_dist=pore_size_dist,
        throat_size_dist=throat_size_dist,
        correlate_pore_throat=correlate_pore_throat,
        beta_params=beta_params,
        seed=seed,
        save_files=False,
    )
    timings['build'] = perf_counter() - t0

    t0 = perf_counter()
    repairs = conservative_geometry_repairs(pn)
    timings['repairs'] = perf_counter() - t0

    t0 = perf_counter()
    try:
        air = op.phase.Air(network=pn)
        air.add_model_collection(op.models.collections.physics.basic)
        air.regenerate_models()
        physics_mode = 'Air'
    except Exception:  # pragma: no cover - fallback path
        air = op.phase.Phase(network=pn)
        air['pore.diffusivity'] = np.full(pn.Np, 1e-5)
        physics_mode = 'Phase'
    timings['phase'] = perf_counter() - t0

    t0 = perf_counter()
    cond_summary = compute_robust_conductance(pn, air)
    timings['conductance'] = perf_counter() - t0

    t0 = perf_counter()
    inlet, outlet = find_inlet_outlet_pores(pn, thickness_fraction=thickness_fraction)
    inlet_r, outlet_r, _ = cluster_restrict(pn, air, inlet, outlet)
    timings['boundary_selection'] = perf_counter() - t0

    t0 = perf_counter()
    dir_res = run_dirichlet(pn, air, inlet_r, outlet_r)
    timings['dirichlet'] = perf_counter() - t0

    t0 = perf_counter()
    neu_res = run_neumann(pn, air, inlet_r, outlet_r, total_rate=rate)
    timings['neumann'] = perf_counter() - t0

    # Derived metrics
    V_p = np.sum((np.pi / 6.0) * pn['pore.diameter']**3)
    td_for_vol = pn['throat.diameter']
    if 'throat.diameter_fixed' in pn.keys():
        td_for_vol = pn['throat.diameter_fixed']
    tl_for_vol = pn['throat.length']
    if 'throat.length_fixed' in pn.keys():
        tl_for_vol = pn['throat.length_fixed']
    V_t = np.sum(np.pi * (td_for_vol / 2.0)**2 * tl_for_vol)
    por = (V_p + V_t) / (L ** 3)

    D_AB = float(air['pore.diffusivity'][0]) if 'pore.diffusivity' in air.keys() else 1e-5
    tau_dir = por * D_AB / dir_res['D_eff'] if dir_res['D_eff'] > 0 else math.inf
    tau_neu = por * D_AB / neu_res['D_eff'] if neu_res['D_eff'] > 0 else math.inf

    total_time = perf_counter() - start_total
    timings['total'] = total_time

    return {
        'timestamp': datetime.utcnow().isoformat(),
        'seed': seed,
        'L_m': L,
        'Np': pn.Np,
        'Nt': pn.Nt,
        'td_fixed_count': repairs['td_fixed_count'],
        'tl_fixed_count': repairs['tl_fixed_count'],
        'cond_n_throats': cond_summary['n_throats'],
        'cond_n_positive': cond_summary['n_positive'],
        'dir_rate': dir_res['rate'],
        'dir_D_eff': dir_res['D_eff'],
        'neu_rate_imposed': rate,
        'neu_rate_calc': neu_res['rate_calc'],
        'neu_D_eff': neu_res['D_eff'],
        'porosity': por,
        'tau_dir': tau_dir,
        'tau_neu': tau_neu,
        'density': density,
        'physics_mode': physics_mode,
        'timings': timings,
    }


def main():
    pilot_L = [
        0.0025118864315095794,
        0.001584893192461114,
        0.001,
    ]
    base_seed = 900
    density = 1e12
    out_csv = FINAL_SET / 'pilot_const_density_1e12.csv'
    if out_csv.exists():
        out_csv.unlink()

    header = ['timestamp', 'seed', 'L_m', 'Np', 'Nt', 'td_fixed_count', 'tl_fixed_count',
              'cond_n_throats', 'cond_n_positive',
              'dir_rate', 'dir_D_eff', 'neu_rate_imposed', 'neu_rate_calc', 'neu_D_eff',
              'porosity', 'tau_dir', 'tau_neu', 'density', 'physics_mode',
              't_build', 't_repairs', 't_phase', 't_conductance', 't_boundary', 't_dirichlet', 't_neumann', 't_total']

    with out_csv.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for idx, L in enumerate(pilot_L):
            seed = base_seed + idx
            print(f"Running pilot for L={L:.6f} m, seed={seed} ...")
            result = run_single(L, seed, density=density)
            timings = result.pop('timings')
            row = {**result,
                   't_build': timings.get('build', 0.0),
                   't_repairs': timings.get('repairs', 0.0),
                   't_phase': timings.get('phase', 0.0),
                   't_conductance': timings.get('conductance', 0.0),
                   't_boundary': timings.get('boundary_selection', 0.0),
                   't_dirichlet': timings.get('dirichlet', 0.0),
                   't_neumann': timings.get('neumann', 0.0),
                   't_total': timings.get('total', 0.0)}
            writer.writerow(row)
            print(f"  -> Np={row['Np']}, Nt={row['Nt']}, total time {row['t_total']:.1f} s")

    print(f"Pilot results written to {out_csv}")


if __name__ == '__main__':
    main()
