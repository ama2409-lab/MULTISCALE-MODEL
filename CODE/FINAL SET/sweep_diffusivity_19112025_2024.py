"""Sweep pore diffusivity values to test D_eff response for Dirichlet and Neumann runs.

This script re-uses the same helper functions/patterns as `sweep_rev_sizes.py` but loops
over a log-spaced diffusivity range from 1e-8 to 1e-2 (default) and repeats each
configuration with multiple random seeds per domain size (default 8).

Output CSV default: `sweep_diffusivity_19112025_2024.csv` (appends rows).

Usage (PowerShell):
    python sweep_diffusivity_19112025_2024.py --sizes 50e-6,100e-6 --density 1e13 --base-seed 100 --seeds-per-size 8 --n-diff 7

Notes:
- Requires the same dependencies as the other scripts (OpenPNM, scipy, numpy).
- For quick smoke tests use `--seeds-per-size 1 --n-diff 3`.
"""
import argparse
import csv
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')

import openpnm as op
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import Counter

from NETWORK_GENERATION_FINAL import build_voronoi_network


# Copied helper functions from sweep_rev_sizes pattern

def conservative_geometry_repairs(network, frac=0.1, minval=1e-9):
    td = network['throat.diameter'].copy()
    bad_td = ~np.isfinite(td) | (td <= 0)
    if np.any(bad_td):
        mean_pos_td = np.mean(td[td > 0]) if np.any(td > 0) else minval
        td_fixed = td.copy()
        td_fixed[bad_td] = max(mean_pos_td * frac, minval)
        network['throat.diameter_fixed'] = td_fixed
    else:
        network['throat.diameter_fixed'] = td

    tl = network['throat.length'].copy()
    bad_tl = ~np.isfinite(tl) | (tl <= 0)
    if np.any(bad_tl):
        mean_pos_tl = np.mean(tl[tl > 0]) if np.any(tl > 0) else minval
        tl_fixed = tl.copy()
        tl_fixed[bad_tl] = max(mean_pos_tl * frac, minval)
        network['throat.length_fixed'] = tl_fixed
    else:
        network['throat.length_fixed'] = tl

    return {'td_fixed_count': int(np.sum(bad_td)), 'tl_fixed_count': int(np.sum(bad_tl))}


def compute_robust_conductance(network, phase, floor=1e-12):
    conns = network['throat.conns']
    td = network['throat.diameter_fixed'] if 'throat.diameter_fixed' in network.keys() else network['throat.diameter']
    tl = network['throat.length_fixed'] if 'throat.length_fixed' in network.keys() else network['throat.length']
    pore_diff = phase['pore.diffusivity']
    D_mean = np.mean(np.vstack([pore_diff[conns[:, 0]], pore_diff[conns[:, 1]]]), axis=0)
    if not np.all(np.isfinite(D_mean)):
        D_mean[~np.isfinite(D_mean)] = np.nanmean(D_mean[np.isfinite(D_mean)])
    area = np.pi * (td / 2.0)**2
    g = D_mean * area / tl
    g = np.where(np.isfinite(g) & (g > 0), g, floor)
    phase['throat.diffusive_conductance'] = g
    return {'n_throats': int(g.size), 'n_positive': int(np.sum(g > 0))}


def find_inlet_outlet_pores(network, axis=0, thickness_fraction=0.05):
    coords = network['pore.coords']
    L = coords[:, axis].max() - coords[:, axis].min()
    slice_thickness = float(thickness_fraction) * float(L)
    minc, maxc = coords[:, axis].min(), coords[:, axis].max()
    inlet = np.where(coords[:, axis] < minc + slice_thickness)[0]
    outlet = np.where(coords[:, axis] > maxc - slice_thickness)[0]
    return inlet, outlet


def cluster_restrict(network, phase, inlet_pores, outlet_pores, eps_g=1e-12):
    conductive_mask = phase['throat.diffusive_conductance'] > eps_g
    conns = network['throat.conns'][conductive_mask]
    if conns.size == 0:
        return inlet_pores, outlet_pores, np.arange(network.Np)
    rows = np.concatenate([conns[:, 0], conns[:, 1]])
    cols = np.concatenate([conns[:, 1], conns[:, 0]])
    data = np.ones(rows.shape[0])
    G = csr_matrix((data, (rows, cols)), shape=(network.Np, network.Np))
    ncomp, labels = connected_components(G, directed=False, connection='weak')
    labels_at_inlet = labels[inlet_pores]
    if len(labels_at_inlet) == 0:
        return inlet_pores, outlet_pores, np.arange(network.Np)
    main_label = Counter(labels_at_inlet).most_common(1)[0][0]
    pores_to_solve = np.where(labels == main_label)[0]
    inlet_in_cluster = np.intersect1d(inlet_pores, pores_to_solve)
    outlet_in_cluster = np.intersect1d(outlet_pores, pores_to_solve)
    if outlet_in_cluster.size == 0 or inlet_in_cluster.size == 0:
        return inlet_pores, outlet_pores, np.union1d(inlet_pores, outlet_pores)
    return inlet_in_cluster, outlet_in_cluster, pores_to_solve


def run_dirichlet(network, phase, inlet, outlet, C_in=1.0, C_out=0.0):
    alg = op.algorithms.FickianDiffusion(network=network, phase=phase)
    alg.settings['solver_family'] = 'pypardiso'
    alg.set_value_BC(pores=inlet, values=C_in, mode='overwrite')
    alg.set_value_BC(pores=outlet, values=C_out, mode='overwrite')
    alg.run()
    rate = alg.rate(pores=inlet).sum()
    coords = network['pore.coords']
    L = coords[:, 0].max() - coords[:, 0].min()
    A = (coords[:, 1].max() - coords[:, 1].min()) * (coords[:, 2].max() - coords[:, 2].min())
    D_eff = rate * L / (A * (C_in - C_out)) if abs(C_in - C_out) > 0 else 0.0
    conc = alg['pore.concentration']
    return {'rate': float(rate), 'D_eff': float(D_eff), 'conc': conc}


def run_neumann(network, phase, inlet, outlet, total_rate=1e-12, C_out=0.0):
    alg = op.algorithms.FickianDiffusion(network=network, phase=phase)
    alg.settings['solver_family'] = 'pypardiso'
    n_in = inlet.size
    if n_in == 0:
        raise RuntimeError('No inlet pores')
    per = float(total_rate) / float(n_in)
    alg.set_rate_BC(pores=inlet, rates=np.full(n_in, per), mode='overwrite')
    alg.set_value_BC(pores=outlet, values=C_out, mode='overwrite')
    alg.run()
    rate_calc = alg.rate(pores=inlet).sum()
    conc = alg['pore.concentration']
    C_in_avg = np.mean(conc[inlet])
    coords = network['pore.coords']
    L = coords[:, 0].max() - coords[:, 0].min()
    A = (coords[:, 1].max() - coords[:, 1].min()) * (coords[:, 2].max() - coords[:, 2].min())
    D_eff = float(total_rate) * L / (A * (C_in_avg - C_out)) if abs(C_in_avg - C_out) > 0 else 0.0
    return {'rate_imposed': float(total_rate), 'rate_calc': float(rate_calc), 'D_eff': float(D_eff), 'C_in_avg': float(C_in_avg), 'conc': conc}


def write_csv_row(path, header, row):
    exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sizes', required=True, help='Comma-separated cubic domain lengths in meters, e.g. 50e-6,100e-6')
    p.add_argument('--density', type=float, default=1e13, help='Pore density (pores per m^3)')
    p.add_argument('--base-seed', type=int, default=42)
    p.add_argument('--seeds-per-size', type=int, default=8, help='Number of independent seeds per domain size')
    p.add_argument('--n-diff', type=int, default=7, help='Number of diffusivity values (logspace between 1e-8 and 1e-2)')
    p.add_argument('--rate', type=float, default=1e-12, help='Total rate for Neumann BC')
    p.add_argument('--out', default='sweep_diffusivity_19112025_2024.csv')
    p.add_argument('--thickness-fraction', type=float, default=0.05, help='Fraction of L used to select inlet/outlet slices (e.g. 0.05)')
    args = p.parse_args()

    sizes = [float(s) for s in args.sizes.split(',')]
    diffs = np.logspace(-8, -2, num=args.n_diff)

    header = ['timestamp', 'seed', 'L_m', 'Np', 'Nt', 'td_fixed_count', 'tl_fixed_count',
              'cond_n_throats', 'cond_n_positive', 'diffusivity',
              'dir_rate', 'dir_D_eff', 'neu_rate_imposed', 'neu_rate_calc', 'neu_D_eff',
              'porosity', 'tau_dir', 'tau_neu', 'density', 'error']

    for i, L in enumerate(sizes):
        for sidx in range(args.seeds_per_size):
            seed = int(args.base_seed + i * args.seeds_per_size + sidx)
            try:
                timestamp = datetime.utcnow().isoformat()
                num_points = int(max(10, round(args.density * (L ** 3))))
                pn = build_voronoi_network(domain_size=(L, L, L), pore_density=args.density, points=num_points,
                                           pore_size_dist={'name': 'lognorm', 's': 0.25, 'scale': 20e-6},
                                           throat_size_dist={'name': 'weibull_min', 'c': 2.5, 'scale': 8e-6},
                                           correlate_pore_throat=False, beta_params=(5.0, 2.0),
                                           seed=seed, save_files=False)
                print(f'Built network L={L:.2e} m (seed={seed}): Np={pn.Np}, Nt={pn.Nt}')

                repairs = conservative_geometry_repairs(pn)
                air = op.phase.Air(network=pn)
                air.add_model_collection(op.models.collections.physics.basic)
                air.regenerate_models()

                # Loop over diffusivities and run
                for diff in diffs:
                    # Override pore diffusivity with a constant field for this run
                    air['pore.diffusivity'] = np.full(pn.Np, float(diff))
                    cond_summary = compute_robust_conductance(pn, air)

                    inlet, outlet = find_inlet_outlet_pores(pn, thickness_fraction=args.thickness_fraction)
                    inlet_r, outlet_r, pores_solve = cluster_restrict(pn, air, inlet, outlet)

                    # Porosity
                    V_p = np.sum((np.pi / 6.0) * pn['pore.diameter']**3)
                    td_for_vol = pn['throat.diameter_fixed'] if 'throat.diameter_fixed' in pn.keys() else pn['throat.diameter']
                    tl_for_vol = pn['throat.length_fixed'] if 'throat.length_fixed' in pn.keys() else pn['throat.length']
                    V_t = np.sum(np.pi * (td_for_vol / 2.0)**2 * tl_for_vol)
                    por = (V_p + V_t) / (L ** 3)

                    # Dirichlet
                    dir_res = run_dirichlet(pn, air, inlet_r, outlet_r)
                    # Neumann
                    neu_res = run_neumann(pn, air, inlet_r, outlet_r, total_rate=args.rate)

                    D_AB = float(air['pore.diffusivity'][0])
                    tau_dir = por * D_AB / dir_res['D_eff'] if dir_res['D_eff'] > 0 else float('inf')
                    tau_neu = por * D_AB / neu_res['D_eff'] if neu_res['D_eff'] > 0 else float('inf')

                    row = {
                        'timestamp': timestamp,
                        'seed': seed,
                        'L_m': L,
                        'Np': pn.Np,
                        'Nt': pn.Nt,
                        'td_fixed_count': repairs['td_fixed_count'],
                        'tl_fixed_count': repairs['tl_fixed_count'],
                        'cond_n_throats': cond_summary['n_throats'],
                        'cond_n_positive': cond_summary['n_positive'],
                        'diffusivity': float(diff),
                        'dir_rate': dir_res['rate'],
                        'dir_D_eff': dir_res['D_eff'],
                        'neu_rate_imposed': neu_res['rate_imposed'],
                        'neu_rate_calc': neu_res['rate_calc'],
                        'neu_D_eff': neu_res['D_eff'],
                        'porosity': por,
                        'tau_dir': tau_dir,
                        'tau_neu': tau_neu,
                        'density': args.density,
                        'error': ''
                    }
                    write_csv_row(args.out, header, row)
                    print(f'Wrote results L={L:.2e}, seed={seed}, diff={diff:.2e} to {args.out}')

            except Exception as e:
                tb = traceback.format_exc()
                print(f'Error at L={L}, seed={seed}: {e}\n{tb}')
                row = {'timestamp': datetime.utcnow().isoformat(), 'seed': seed, 'L_m': L, 'Np': '', 'Nt': '',
                       'td_fixed_count': '', 'tl_fixed_count': '', 'cond_n_throats': '', 'cond_n_positive': '',
                       'diffusivity': '', 'dir_rate': '', 'dir_D_eff': '', 'neu_rate_imposed': '', 'neu_rate_calc': '', 'neu_D_eff': '',
                       'porosity': '', 'tau_dir': '', 'tau_neu': '', 'density': args.density, 'error': str(e)}
                write_csv_row(args.out, header, row)


if __name__ == '__main__':
    main()
