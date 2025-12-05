# works!! - NEUMANN (UPPER-BOUND) VERSION
import openpnm as op
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import Counter

op.visualization.set_mpl_style()
print("--- Voronoi Fickian Diffusion (robust version) [NEUMANN BC] ---")

# =============================================================================
# 1. NETWORK SETUP
# =============================================================================
SEED = 42
from NETWORK_GENERATION_FINAL import build_voronoi_network
domain = (250e-6, 250e-6, 250e-6)
density = 1e13
num_points = int(max(10, round(density * np.prod(domain))))
pn = build_voronoi_network(domain_size=domain, pore_density=density, points=num_points,
                           pore_size_dist={'name': 'lognorm', 's': 0.25, 'scale': 20e-6},
                           throat_size_dist={'name': 'weibull_min', 'c': 2.5, 'scale': 8e-6},
                           correlate_pore_throat=False, beta_params=(5.0, 2.0),
                           seed=SEED, save_files=False)
print(f"Using Voronoi network with {pn.Np} pores and {pn.Nt} throats.")

# Quick health check (non-fatal)
op.utils.check_network_health(pn)

# Fail fast if essential arrays are missing
if any(key not in pn for key in ['pore.coords', 'throat.conns', 'pore.diameter', 'throat.diameter']):
    raise RuntimeError("Essential network arrays are missing.")

# =============================================================================
# 2. GEOMETRY REPAIRS & PHYSICS SETUP
# =============================================================================
air = op.phase.Air(network=pn)
phys = op.models.collections.physics.basic
try:
    del phys['throat.entry_pressure']
except Exception:
    pass
air.add_model_collection(phys)
air.regenerate_models()

# --- CONSERVATIVE REPAIRS (prevent zero/NaN geometry that kills conductances) ---
print('Running conservative geometry/volume repairs (if needed)')
# Repair throat.diameter
td = np.asarray(pn['throat.diameter'], dtype=float)
bad_td = (~np.isfinite(td)) | (td <= 0)
if np.any(bad_td):
    pos = td[td > 0]
    if pos.size > 0:
        mean_pos_td = float(np.mean(pos))
    else:
        mean_pos_td = 1e-9
    replacement_td = max(mean_pos_td * 0.1, 1e-9)
    td[bad_td] = replacement_td
    pn['throat.diameter'] = td
    print(f"Fixed {int(np.sum(bad_td))} throat.diameter entries -> {replacement_td:.2e} m")

# Repair throat.length
tl = np.asarray(pn['throat.length'], dtype=float)
bad_tl = (~np.isfinite(tl)) | (tl <= 0)
if np.any(bad_tl):
    posl = tl[tl > 0]
    if posl.size > 0:
        mean_pos_tl = float(np.mean(posl))
    else:
        coords = np.asarray(pn['pore.coords'])
        conns = np.asarray(pn['throat.conns'])
        dists = np.linalg.norm(coords[conns[:, 0]] - coords[conns[:, 1]], axis=1)
        mean_pos_tl = float(np.mean(dists[dists > 0])) if np.any(dists > 0) else 1e-9
    replacement_tl = max(mean_pos_tl * 0.1, 1e-9)
    tl[bad_tl] = replacement_tl
    pn['throat.length'] = tl
    print(f"Fixed {int(np.sum(bad_tl))} throat.length entries -> {replacement_tl:.2e} m")

# Repair pore/throat volumes if NaN
pv = np.asarray(pn.get('pore.volume', np.array([])), dtype=float)
if pv.size == 0 or not np.all(np.isfinite(pv)):
    try:
        pd = np.asarray(pn['pore.diameter'], dtype=float)
        pv_new = (np.pi / 6.0) * pd ** 3
        pn['pore.volume'] = pv_new
        print('Recomputed pore.volume from pore.diameter')
    except Exception:
        pass
tv = np.asarray(pn.get('throat.volume', np.array([])), dtype=float)
if tv.size == 0 or not np.all(np.isfinite(tv)):
    try:
        td = np.asarray(pn['throat.diameter'], dtype=float)
        tl = np.asarray(pn['throat.length'], dtype=float)
        tv_new = np.pi * (td / 2.0) ** 2 * tl
        pn['throat.volume'] = tv_new
        print('Recomputed throat.volume from diameter and length')
    except Exception:
        pass

# Regenerate models after manual edits
try:
    pn.regenerate_models()
except Exception:
    pass
try:
    air.regenerate_models()
except Exception:
    pass

# If conductances are still missing/zero, compute a physics-based fallback per throat
g_test = np.asarray(air.get('throat.diffusive_conductance', np.array([])), dtype=float)
bad_g = (g_test.size == 0) or (~np.isfinite(g_test)).any() or (g_test <= 0).sum() > 0
if bad_g:
    print('Computing fallback throat.diffusive_conductance for any non-positive entries')
    g = np.asarray(air.get('throat.diffusive_conductance', np.zeros(pn.Nt)), dtype=float)
    pore_diff = np.asarray(air.get('pore.diffusivity', np.full(pn.Np, np.nan)), dtype=float)
    conns_all = np.asarray(pn['throat.conns'])
    for i in range(pn.Nt):
        if not np.isfinite(g[i]) or g[i] <= 0:
            a, b = conns_all[i]
            D_local = np.nanmean([pore_diff[a] if np.isfinite(pore_diff[a]) else np.nan,
                                  pore_diff[b] if np.isfinite(pore_diff[b]) else np.nan])
            if not np.isfinite(D_local):
                D_local = 1e-9
            d_th = float(pn['throat.diameter'][i])
            l_th = float(pn['throat.length'][i]) if float(pn['throat.length'][i]) > 0 else 1e-9
            area = np.pi * (d_th / 2.0) ** 2
            g[i] = D_local * area / l_th
    air['throat.diffusive_conductance'] = g
    print('Fallback conductances written to phase')
    try:
        print('Recomputing conductances for all throats to ensure connectivity...')
        pore_diff = np.asarray(air.get('pore.diffusivity', np.full(pn.Np, np.nan)), dtype=float)
        conns_all = np.asarray(pn['throat.conns'])
        td_all = np.asarray(pn['throat.diameter'], dtype=float)
        tl_all = np.asarray(pn['throat.length'], dtype=float)
        D_mean = np.nanmean(np.vstack([pore_diff[conns_all[:, 0]], pore_diff[conns_all[:, 1]]]), axis=0)
        D_mean[~np.isfinite(D_mean)] = 1e-9
        area = np.pi * (td_all / 2.0) ** 2
        tl_safe = np.copy(tl_all)
        tl_safe[~np.isfinite(tl_safe) | (tl_safe <= 0)] = 1e-9
        g_all_new = D_mean * area / tl_safe
        eps_g = 1e-12
        g_all_new = np.where(np.isfinite(g_all_new) & (g_all_new > 0), g_all_new, eps_g)
        air['throat.diffusive_conductance'] = g_all_new
        print('Wrote robust conductances for all throats (with small positive floor)')
    except Exception:
        pass

# 3) Setup Fickian diffusion
fd = op.algorithms.FickianDiffusion(network=pn, phase=air)

# --- Find pores on ALL faces ---
inlet_pores = pn.pores('left_boundary')
outlet_pores = pn.pores('right_boundary')
top_pores = pn.pores('top_boundary')
bottom_pores = pn.pores('bottom_boundary')
front_pores = pn.pores('front_boundary')
back_pores = pn.pores('back_boundary')

side_pores = np.unique(np.concatenate((top_pores, bottom_pores, front_pores, back_pores)))

# --- Fallback (if labels are missing) ---
if inlet_pores.size == 0 or outlet_pores.size == 0:
    print("Fallback: finding inlet/outlet pores by coordinates...")
    coords = pn['pore.coords']
    axis = 0
    mean_pdia = pn['pore.diameter'].mean()
    slice_thickness = 3 * mean_pdia
    minc = coords[:, axis].min(); maxc = coords[:, axis].max()
    inlet_pores = np.where(coords[:, axis] < minc + slice_thickness)[0]
    outlet_pores = np.where(coords[:, axis] > maxc - slice_thickness)[0]

if inlet_pores.size == 0 or outlet_pores.size == 0:
    raise RuntimeError('Could not find inlet/outlet pores')

# --- Define Neumann BC parameters ---
rate_in_set = 1e-12  # mol/s (this is an arbitrary value)
C_out_set = 0.0        # Reference concentration

# Apply the Neumann (rate) BC at the inlet
fd.set_rate_BC(pores=inlet_pores, rates=rate_in_set)
# Apply the Dirichlet (value) BC at the outlet
fd.set_value_BC(pores=outlet_pores, values=C_out_set)
print(f"Applied Neumann BC: rate={rate_in_set:.2e} mol/s on {len(inlet_pores)} inlet pores")
print(f"Applied Dirichlet BC: C={C_out_set} on {len(outlet_pores)} outlet pores")

# Apply no-flux (zero-rate) to all side walls
if side_pores.size > 0:
    fd.set_rate_BC(pores=side_pores, rates=0.0)
    print(f"Applied Neumann (zero-rate) BC to {len(side_pores)} side pores")


# 4) Clean conductances: replace non-finite with zero
g = air['throat.diffusive_conductance']
g = np.asarray(g, dtype=float)
g[~np.isfinite(g)] = 0.0
air['throat.diffusive_conductance'] = g

# 5) Identify conductive cluster using SciPy connected_components on conductive throats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import Counter

conductive_mask = air['throat.diffusive_conductance'] > 0
conns = pn['throat.conns'][conductive_mask]
if conns.size == 0:
    print('Warning: No conductive throats (all conductances zero). Proceeding with BCs on labeled boundaries.')
    pores_to_solve = np.arange(pn.Np)
else:
    rows = np.concatenate([conns[:, 0], conns[:, 1]])
    cols = np.concatenate([conns[:, 1], conns[:, 0]])
    data = np.ones(rows.size, dtype=int)
    G = csr_matrix((data, (rows, cols)), shape=(pn.Np, pn.Np))
    ncomp, p_labels = connected_components(G, directed=False, connection='weak')
    labels_at_inlet = p_labels[inlet_pores]
    cnt = Counter(labels_at_inlet)
    if len(cnt) == 0:
        print('Warning: conductive cluster does not include inlet pores; falling back to all pores')
        pores_to_solve = np.arange(pn.Np)
    else:
        main_label = int(cnt.most_common(1)[0][0])
        pores_to_solve = np.where(p_labels == main_label)[0]
        print(f'Main conductive cluster label {main_label} with {len(pores_to_solve)} pores')

# Restrict BCs to cluster if possible, else keep original label-based BCs
inlet_in_cluster = np.intersect1d(inlet_pores, pores_to_solve)
outlet_in_cluster = np.intersect1d(outlet_pores, pores_to_solve)
if inlet_in_cluster.size == 0 or outlet_in_cluster.size == 0:
    print('Warning: inlet/outlet not found inside main cluster; using original labeled boundary pores')
    inlet_in_cluster = inlet_pores
    outlet_in_cluster = outlet_pores

# --- Overwrite BCs for the identified cluster ---
fd.set_rate_BC(pores=inlet_in_cluster, rates=rate_in_set, mode='overwrite')
fd.set_value_BC(pores=outlet_in_cluster, values=C_out_set, mode='overwrite')


# 6) Pin isolated pores (zero incident conductance) to avoid singular matrix
g_all = air['throat.diffusive_conductance']
conns_all = pn['throat.conns']
try:
    g_arr = np.asarray(g_all, dtype=float)
    print('Conductance array: finite=', np.sum(np.isfinite(g_arr)), 'positive=', np.sum(g_arr>0), 'min=', np.nanmin(g_arr), 'max=', np.nanmax(g_arr))
    if outlet_pores.size>0:
        sample_out = int(outlet_pores[0])
        touching = np.where((conns_all[:,0]==sample_out)|(conns_all[:,1]==sample_out))[0]
        print(f'Thorates touching outlet pore {sample_out}: indices={touching[:10]} (show up to 10)')
        print('their g values:', g_arr[touching[:10]])
except Exception:
    print('Could not summarize conductance array for debug')
incident = np.zeros(pn.Np, dtype=float)
for i, (a, b) in enumerate(conns_all):
    incident[a] += float(g_all[i])
    incident[b] += float(g_all[i])
isolated = np.where(incident == 0)[0]
if isolated.size > 0:
    # Pinning isolated pores to C_out_set (0.0) is safer than pinning to C_in
    print(f'Pinning {isolated.size} isolated pores to C_out_set to avoid singular matrix')
    fd.set_value_BC(pores=isolated, values=C_out_set, mode='overwrite')

# 7) Run solver
print('Running FickianDiffusion solver...')
fd.run()
print('Solver finished')

# DEBUG: print concentration stats to diagnose zero flow
try:
    conc = fd['pore.concentration']
    print('Concentration stats: min={:.6e}, max={:.6e}, mean={:.6e}'.format(float(np.nanmin(conc)), float(np.nanmax(conc)), float(np.nanmean(conc))))
    print('Sample inlet concentrations:', conc[inlet_in_cluster][:10])
    print('Sample outlet concentrations:', conc[outlet_in_cluster][:10])
except Exception as _:
    print('Could not read pore.concentration for debug')

# 8) Post-process
# The inlet rate is the one we SET, not one we calculate.
rate_in = rate_in_set
print(f'Molar flow rate (set): {rate_in:.5e} mol/s')

# We need to find the AVERAGE concentration at the inlet pores
# to calculate the concentration gradient
conc = fd['pore.concentration']
C_in_avg = np.mean(conc[inlet_in_cluster])
# The outlet concentration is the value we set
C_out_val = C_out_set

print(f'Calculated avg. inlet C: {C_in_avg:.5e} mol/m3')
print(f'Set outlet C: {C_out_val:.5e} mol/m3')

# Use user-provided domain for geometry
L = domain[0]
A = domain[1] * domain[2]
delta_C = C_in_avg - C_out_val

D_eff = rate_in * L / (A * delta_C) if delta_C != 0 else 0.0
print('Effective diffusivity: {0:.6E} m2/s'.format(D_eff))


# Robust porosity calculation from diameters/lengths (avoid possible NaNs in stored arrays)
pd = np.asarray(pn.get('pore.diameter', np.zeros(pn.Np)), dtype=float)
td = np.asarray(pn.get('throat.diameter', np.zeros(pn.Nt)), dtype=float)
tl = np.asarray(pn.get('throat.length', np.ones(pn.Nt)), dtype=float)
pv = (np.pi / 6.0) * pd ** 3
tv = np.pi * (td / 2.0) ** 2 * tl
V_p = float(np.nansum(pv))
V_t = float(np.nansum(tv))
V_bulk = float(np.prod(domain))
por = (V_p + V_t) / V_bulk if V_bulk > 0 else float('nan')
D_AB = float(air['pore.diffusivity'][0]) if 'pore.diffusivity' in air.keys() else 1e-5
tau = por * D_AB / D_eff if D_eff != 0 else float('inf')
print('Porosity: {0:.6E}'.format(por) if np.isfinite(por) else f'Porosity: {por}')
print('Tortuosity: {0:.6E}'.format(tau) if np.isfinite(tau) else f'Tortuosity: {tau}')

# If rate_in is zero (solver returned zero flux), compute flux directly from throat conductances
if abs(rate_in) < 1e-30:
    print('fd.rate returned zero; computing throat-flux-based inlet rate as fallback')
    conc = fd['pore.concentration']
    conns_all = np.asarray(pn['throat.conns'])
    g_all = np.asarray(air['throat.diffusive_conductance'], dtype=float)
    inlet_set = set(inlet_in_cluster.tolist())
    throat_idxs = np.where(np.logical_xor(np.isin(conns_all[:, 0], list(inlet_set)), np.isin(conns_all[:, 1], list(inlet_set))))[0]
    fluxes = []
    for i in throat_idxs:
        a, b = conns_all[i]
        if a in inlet_set and b not in inlet_set:
            flux = g_all[i] * (conc[a] - conc[b])
        elif b in inlet_set and a not in inlet_set:
            flux = g_all[i] * (conc[b] - conc[a])
        else:
            continue
        fluxes.append(float(flux))
    alt_rate = float(np.nansum(fluxes)) if len(fluxes) > 0 else 0.0
    print(f'Alternate throat-based inlet rate: {alt_rate:.5e} mol/s (summed over {len(fluxes)} throats)')
    if alt_rate > 0:
        rate_in = alt_rate
        D_eff = rate_in * L / (A * (delta_C)) if delta_C != 0 else 0.0
        tau = por * D_AB / D_eff if D_eff != 0 else float('inf')
        print('Updated Effective diffusivity (from throat flux): {0:.6E} m2/s'.format(D_eff))
        print('Updated Tortuosity: {0:.6E}'.format(tau) if np.isfinite(tau) else f'Updated Tortuosity: {tau}')

# Save plot
try:
    fig, ax = plt.subplots(figsize=[8, 6])
    conc = fd['pore.concentration']
    op.visualization.plot_coordinates(pn, ax=ax, color_by=conc, size_by=pn['pore.diameter'], markersize=200)
    op.visualization.plot_connections(pn, ax=ax, color_by=fd.interpolate_data('throat.concentration'), linewidth=2)
    ax.set_axis_off()
    fig.tight_layout()
    out = 'fickian_voronoi_concentration_neumann.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved concentration plot to {out}')
except Exception as e:
    print('Could not save plot:', e)

print('Voronoi diffusion run complete')