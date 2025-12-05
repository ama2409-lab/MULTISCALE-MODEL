# NEUMANN - modified from Dirichlet
import openpnm as op
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import Counter

op.visualization.set_mpl_style()
print("--- Voronoi Fickian Diffusion NEUMANN (robust and clean version) ---")

# Minimal CLI so rate/seed can be overridden without changing the file
import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--rate', type=float, default=1e-12)
args, _ = parser.parse_known_args()
SEED = args.seed

# =============================================================================
# 1. NETWORK SETUP
# =============================================================================
SEED = 42
# Assuming NETWORK_GENERATION_FINAL.py is available in the same directory
from NETWORK_GENERATION_FINAL import build_voronoi_network, DEFAULT_DOMAIN, DEFAULT_DENSITY

# Allow domain/density override from CLI (seed/rate already supported)
import argparse as _arg
_parser = _arg.ArgumentParser(add_help=False)
_parser.add_argument('--domain', type=str, default=None)
_parser.add_argument('--density', type=float, default=None)
_args, _ = _parser.parse_known_args()

if _args.domain:
    parts = [float(x) for x in _args.domain.split(',')]
    if len(parts) == 1:
        domain = (parts[0], parts[0], parts[0])
    elif len(parts) == 3:
        domain = tuple(parts)
    else:
        raise ValueError('Invalid --domain format')
else:
    domain = DEFAULT_DOMAIN

density = _args.density if _args.density is not None else DEFAULT_DENSITY
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

# --- MODIFICATION: Set Neumann (Rate) BC at inlet, Dirichlet (Value) at outlet ---
rate_in_imposed = args.rate  # mol/s (total rate to inject for Neumann run)
C_out = 0.0             # Still need a reference concentration

# NOTE: Neumann (rate) BC will be applied after cluster detection so we can
# distribute the total rate across the actual inlet pores in the conductive cluster.
# No BCs are applied here yet.

# --- End Modification ---

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

    # Robust fallbacks: ensure we always have at least one inlet and outlet pore
    if inlet_in_cluster.size == 0:
        if inlet_pores.size > 0:
            inlet_in_cluster = inlet_pores
            print('No inlet pores found in cluster; falling back to raw inlet_pores.')
        else:
            inlet_in_cluster = np.array([np.argmin(coords[:, axis])], dtype=int)
            print('No inlet_pores available; selected pore', inlet_in_cluster[0])

    if outlet_in_cluster.size == 0:
        if outlet_pores.size > 0:
            outlet_in_cluster = outlet_pores
            print('No outlet pores found in cluster; falling back to raw outlet_pores.')
        else:
            outlet_in_cluster = np.array([np.argmax(coords[:, axis])], dtype=int)
            print('No outlet_pores available; selected pore', outlet_in_cluster[0])
else:
    # If no cluster info, fall back to raw inlet/outlet sets
    inlet_in_cluster, outlet_in_cluster = inlet_pores, outlet_pores
    if inlet_in_cluster.size == 0:
        inlet_in_cluster = np.array([np.argmin(coords[:, axis])], dtype=int)
        print('No inlet_pores found by slice; selected pore', inlet_in_cluster[0])
    if outlet_in_cluster.size == 0:
        outlet_in_cluster = np.array([np.argmax(coords[:, axis])], dtype=int)
        print('No outlet_pores found by slice; selected pore', outlet_in_cluster[0])

# Now apply BCs deterministically using the resolved inlet/outlet pore lists
rate_per_pore = float(rate_in_imposed) / float(max(1, inlet_in_cluster.size))
rates_array = np.full(inlet_in_cluster.size, rate_per_pore)
fd.set_rate_BC(pores=inlet_in_cluster, rates=rates_array, mode='overwrite')
fd.set_value_BC(pores=outlet_in_cluster, values=C_out, mode='overwrite')
        



# --- MODIFICATION: Update print statement ---
print(f"Final BCs: Rate={rate_in_imposed:.2e} on {len(inlet_in_cluster)} inlet pores; C={C_out} on {len(outlet_in_cluster)} outlet pores.")
# --- End Modification ---

# =============================================================================
# 4. SOLVE AND POST-PROCESS
# =============================================================================
print('Running FickianDiffusion solver...')
fd.run() #try and implment pypardiso - check how longpython -m cProfile -s tottime your_script_name.py
print('Solver finished.')

conc = fd['pore.concentration']
print(f'Concentration stats: min={np.nanmin(conc):.3e}, max={np.nanmax(conc):.3e}, mean={np.nanmean(conc):.3e}')

# --- MODIFICATION: Update D_eff calculation for Neumann BC ---
# Check the rate calculated by the solver (should match our imposed rate)
rate_in_check = fd.rate(pores=inlet_in_cluster).sum()
print(f'Imposed molar flow rate: {rate_in_imposed:.5e} mol/s (Check: {rate_in_check:.5e})')

# Calculate D_eff based on the *resulting* average inlet concentration
C_in_avg = np.mean(conc[inlet_in_cluster])
print(f'Resulting average inlet concentration: {C_in_avg:.5e}')

L, A = domain[0], domain[1] * domain[2]
# Use the imposed rate and the *resulting* concentration difference
D_eff = rate_in_imposed * L / (A * (C_in_avg - C_out)) if abs(C_in_avg - C_out) > 0 else 0.0
print(f'Effective diffusivity: {D_eff:.6E} m2/s')
# --- End Modification ---

# Robust porosity calculation
V_p = np.sum((np.pi / 6.0) * pn['pore.diameter']**3)
V_t = np.sum(np.pi * (pn['throat.diameter'] / 2.0)**2 * pn['throat.length'])
por = (V_p + V_t) / np.prod(domain)
D_AB = air['pore.diffusivity'][0]
tau = por * D_AB / D_eff if D_eff > 0 else float('inf')
print(f'Porosity: {por:.6E}')
print(f'Tortuosity: {tau:.6E}')

# Fallback rate calculation (still useful as a check)
# --- MODIFICATION: Use rate_in_check variable ---
if abs(rate_in_check) < 1e-30:
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
    # --- MODIFICATION: Update filename ---
    fig.savefig('fickian_voronoi_concentration_neumann.png', dpi=200, bbox_inches='tight')
    # --- End Modification ---
    plt.close(fig)
    print('Saved concentration plot.')
except Exception as e:
    print(f'Could not save plot: {e}')

print('Voronoi diffusion run complete.')