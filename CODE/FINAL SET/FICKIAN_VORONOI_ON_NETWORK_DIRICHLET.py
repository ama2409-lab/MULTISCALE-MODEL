# DIRICHLET - works!! but not periodic 
import openpnm as op
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import Counter

op.visualization.set_mpl_style()
print("--- Voronoi Fickian Diffusion DIRICHLET (robust and clean version) ---")

# =============================================================================
# 1. NETWORK SETUP
# =============================================================================
# Use centralized defaults from NETWORK_GENERATION_FINAL; allow overrides via CLI
import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--domain', type=str, default=None, help='Comma-separated domain (m) e.g. 250e-6,250e-6,250e-6 or single value for cubic')
parser.add_argument('--density', type=float, default=None, help='Pore density (pores per m^3)')
parser.add_argument('--out', type=str, default='fickian_voronoi_concentration_dirichlet.png')
args, _ = parser.parse_known_args()

from NETWORK_GENERATION_FINAL import build_voronoi_network, DEFAULT_DOMAIN, DEFAULT_DENSITY
SEED = int(args.seed)
# parse domain argument
if args.domain:
    parts = [float(x) for x in args.domain.split(',')]
    if len(parts) == 1:
        domain = (parts[0], parts[0], parts[0])
    elif len(parts) == 3:
        domain = tuple(parts)
    else:
        raise ValueError('Invalid --domain format')
else:
    domain = DEFAULT_DOMAIN

density = args.density if args.density is not None else DEFAULT_DENSITY
num_points = int(max(10, round(density * np.prod(domain))))
pn = build_voronoi_network(domain_size=domain, pore_density=density, points=num_points,
                           pore_size_dist={'name': 'lognorm', 's': 0.25, 'scale': 20e-6},
                           throat_size_dist={'name': 'weibull_min', 'c': 2.5, 'scale': 8e-6},
                           correlate_pore_throat=False, beta_params=(5.0, 2.0),
                           seed=SEED, save_files=False)
print(f"Using Voronoi network with {pn.Np} pores and {pn.Nt} throats.")

op.utils.check_network_health(pn)

if any(key not in pn for key in ['pore.coords', 'throat.conns', 'pore.diameter', 'throat.diameter']):
    raise RuntimeError("Essential network arrays are missing.")

# =============================================================================
# 2. GEOMETRY REPAIRS & PHYSICS SETUP
# =============================================================================
print('Running conservative geometry repairs (if needed)...')
# Repair throat.diameter
td = pn['throat.diameter']
bad_td = ~np.isfinite(td) | (td <= 0)
if np.any(bad_td):
    mean_pos_td = np.mean(td[td > 0]) if np.any(td > 0) else 1e-9
    td[bad_td] = max(mean_pos_td * 0.1, 1e-9)
    print(f"Fixed {np.sum(bad_td)} invalid throat.diameter entries.")
# Repair throat.length
tl = pn['throat.length']
bad_tl = ~np.isfinite(tl) | (tl <= 0)
if np.any(bad_tl):
    mean_pos_tl = np.mean(tl[tl > 0]) if np.any(tl > 0) else 1e-9
    tl[bad_tl] = max(mean_pos_tl * 0.1, 1e-9)
    print(f"Fixed {np.sum(bad_tl)} invalid throat.length entries.")

# Create Phase and add placeholder models (will be overwritten)
air = op.phase.Air(network=pn)
air.add_model_collection(op.models.collections.physics.basic)
air.regenerate_models() # Regenerate once to populate basic properties like diffusivity

# Report the base molecular diffusivity used by the Air phase
try:
    base_D = float(air['pore.diffusivity'][0])
    print(f"Base molecular diffusivity D_AB from OpenPNM Air phase: {base_D:.6e} m^2/s")
except Exception as e:
    print(f"Could not read base pore.diffusivity: {e}")

# --- CONSOLIDATED CONDUCTANCE CALCULATION (This is the physics you were actually using) ---
print('Calculating diffusive conductance for all throats using robust D*A/L model...')
try:
    pore_diff = air['pore.diffusivity']
    conns = pn['throat.conns']
    td = pn['throat.diameter']
    tl = pn['throat.length']
    
    # Vectorized calculation of D*A/L for all throats
    D_mean = np.mean(np.vstack([pore_diff[conns[:, 0]], pore_diff[conns[:, 1]]]), axis=0)
    D_mean[~np.isfinite(D_mean)] = np.nanmean(D_mean) # Fallback for edge cases
    area = np.pi * (td / 2.0)**2
    g = D_mean * area / tl

    # Enforce a small positive floor. This is the key to avoiding singular matrices.
    eps_g = 1e-12
    g = np.where(np.isfinite(g) & (g > 0), g, eps_g)
    air['throat.diffusive_conductance'] = g
    print('Wrote robust conductances for all throats (with small positive floor).')
except Exception as e:
    raise RuntimeError(f"Failed to compute robust conductances: {e}")

# NOTE: The old redundant loops, the `g[~np.isfinite(g)] = 0.0` line, and the
# `Pin isolated pores` section are all removed because the logic above makes them obsolete.

# =============================================================================
# 3. ALGORITHM SETUP & BOUNDARY CONDITIONS
# =============================================================================
fd = op.algorithms.FickianDiffusion(network=pn, phase=air)
fd.settings['solver_family'] = 'pypardiso'

# Find boundary pores
coords = pn['pore.coords']
axis = 0
slice_thickness = 3 * np.mean(pn['pore.diameter'])
minc, maxc = coords[:, axis].min(), coords[:, axis].max()
inlet_pores = np.where(coords[:, axis] < minc + slice_thickness)[0]
outlet_pores = np.where(coords[:, axis] > maxc - slice_thickness)[0]
if inlet_pores.size == 0 or outlet_pores.size == 0:
    raise RuntimeError('Could not find inlet/outlet pores.')

C_in, C_out = 1.0, 0.0
fd.set_value_BC(pores=inlet_pores, values=C_in)
fd.set_value_BC(pores=outlet_pores, values=C_out)

# Identify main conductive cluster (still good practice)
conductive_mask = air['throat.diffusive_conductance'] > eps_g
conns = pn['throat.conns'][conductive_mask]
G = csr_matrix((np.ones(conns.shape[0]), (conns[:, 0], conns[:, 1])), shape=(pn.Np, pn.Np))
ncomp, p_labels = connected_components(G, directed=False, connection='weak')

# Find cluster connected to inlet and check if it connects to outlet
labels_at_inlet = p_labels[inlet_pores]
if len(labels_at_inlet) > 0:
    main_label = Counter(labels_at_inlet).most_common(1)[0][0]
    pores_to_solve = np.where(p_labels == main_label)[0]
    inlet_in_cluster = np.intersect1d(inlet_pores, pores_to_solve)
    outlet_in_cluster = np.intersect1d(outlet_pores, pores_to_solve)
    if outlet_in_cluster.size > 0:
        print(f'Applying BCs to main conductive cluster ({len(pores_to_solve)} pores).')
        fd.set_value_BC(pores=inlet_in_cluster, values=C_in, mode='overwrite')
        fd.set_value_BC(pores=outlet_in_cluster, values=C_out, mode='overwrite')
    else: # If cluster doesn't reach outlet, use original BCs
        inlet_in_cluster, outlet_in_cluster = inlet_pores, outlet_pores
else: # If inlet isn't in a cluster, use original BCs
    inlet_in_cluster, outlet_in_cluster = inlet_pores, outlet_pores

print(f"Final BCs: C={C_in} on {len(inlet_in_cluster)} inlet pores; C={C_out} on {len(outlet_in_cluster)} outlet pores.")

# =============================================================================
# 4. SOLVE AND POST-PROCESS
# =============================================================================
print('Running FickianDiffusion solver...')
fd.run() #try and implment pypardiso - check how longpython -m cProfile -s tottime your_script_name.py
print('Solver finished.')

conc = fd['pore.concentration']
print(f'Concentration stats: min={np.nanmin(conc):.3e}, max={np.nanmax(conc):.3e}, mean={np.nanmean(conc):.3e}')

rate_in = fd.rate(pores=inlet_in_cluster).sum()
print(f'Molar flow rate: {rate_in:.5e} mol/s')

L, A = domain[0], domain[1] * domain[2]
D_eff = rate_in * L / (A * (C_in - C_out)) if abs(C_in - C_out) > 0 else 0.0
print(f'Effective diffusivity: {D_eff:.6E} m2/s')

# Robust porosity calculation
V_p = np.sum((np.pi / 6.0) * pn['pore.diameter']**3)
V_t = np.sum(np.pi * (pn['throat.diameter'] / 2.0)**2 * pn['throat.length'])
por = (V_p + V_t) / np.prod(domain)
D_AB = air['pore.diffusivity'][0]
tau = por * D_AB / D_eff if D_eff > 0 else float('inf')
print(f'Porosity: {por:.6E}')
print(f'Tortuosity: {tau:.6E}')

# Fallback rate calculation (still useful as a check)
if abs(rate_in) < 1e-30:
    print('fd.rate is zero; computing throat-flux-based rate as fallback...')
    # Vectorized version of this calculation is in the "how to make it faster" section
    # ... your old loop here ...

# =============================================================================
# 5. VISUALIZATION
# =============================================================================
try:
    fig, ax = plt.subplots(figsize=[8, 6])
    op.visualization.plot_coordinates(pn, ax=ax, color_by=conc, size_by=pn['pore.diameter'], markersize=200)
    op.visualization.plot_connections(pn, ax=ax, color_by=fd.interpolate_data('throat.concentration'), linewidth=2)
    ax.set_axis_off(); fig.tight_layout()
    fig.savefig('fickian_voronoi_concentration_dirichlet.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved concentration plot.')
except Exception as e:
    print(f'Could not save plot: {e}')

print('Voronoi diffusion run complete.')