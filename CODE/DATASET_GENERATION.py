import openpnm as op
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats as spst
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


# Set a consistent style for the plots
op.visualization.set_mpl_style()

# --------------------------------------------------------------------------- #
# --- 1. PORE NETWORK GENERATION (Directly in OpenPNM) ---
# --------------------------------------------------------------------------- #
print("--- 1. PORE NETWORK GENERATION ---")
# Build a Voronoi-derived network using the helper in NETWORKgeneration.py
from NETWORK_GENERATION_FINAL import build_voronoi_network

# Create the network (adjust points and seed as desired)
pn = build_voronoi_network(points=500, domain_size=(250e-6, 250e-6, 250e-6), seed=42, save_png=False)
print(f"Using Voronoi network with {pn.Np} pores and {pn.Nt} throats.")


# Quick health check (non-fatal)
try:
    h = op.utils.check_network_health(pn)
except Exception:
    h = None

# --- STRICT REQUIRED ARRAYS CHECK ---
# Fail fast if essential arrays are missing (user requested strict behavior)
required_arrays = ['pore.coords', 'throat.conns', 'pore.volume', 'throat.volume', 'pore.diameter', 'throat.diameter']
missing = [a for a in required_arrays if a not in pn.keys()]
if len(missing) > 0:
    raise RuntimeError(f"Required network arrays missing: {missing}. Aborting.")



# --------------------------------------------------------------------------- #
# --- 2. DIFFUSION SIMULATION SETUP ---
# --------------------------------------------------------------------------- #
print("\n--- 2. DIFFUSION SIMULATION SETUP ---")

# --- CREATE PHASE AND PHYSICS ---
# A "phase" represents the fluid in the pores (e.g., Air, Water).
# "Physics" defines how the phase properties interact with the network geometry.
print("Creating an 'Air' phase object...")
air = op.phase.Air(network=pn)

print("Adding 'basic' physics models to the Air phase (if missing)...")
phys = op.models.collections.physics.basic
try:
    air.add_model_collection(phys)
    air.regenerate_models()
except Exception:
    # If models already present from the builder, this may be a no-op
    pass
print("Physics models ready (transport properties available).")

# --- SETUP THE ALGORITHM ---
# The algorithm object is what actually solves the equations.
print("\nSetting up the 'FickianDiffusion' algorithm...")
fd = op.algorithms.FickianDiffusion(network=pn, phase=air)

# Identify inlet/outlet pores robustly. Prefer labeled faces, fall back to coords.
#try:
#    inlet = pn.pores('left')
#    outlet = pn.pores('right')
#except Exception:
inlet = np.array([], dtype=int)
outlet = np.array([], dtype=int)




# Fallback to coordinate-based selection along the x-axis (axis=0)
# --- NEW BOUNDARY CONDITIONS: SLICE METHOD ---
print("\nDefining boundary pores using a physically-based slice thickness...")

# 1. Define the axis of flow (0=X, 1=Y, 2=Z)
axis = 0
coords = pn['pore.coords']

# 2. Calculate a slice thickness based on the network's microstructure
# This ensures the boundary condition is physically consistent across different networks.
# A value of 2-5 times the mean pore diameter is a good starting point.
mean_pore_dia = pn['pore.diameter'].mean()
slice_thickness = 3 * mean_pore_dia
print(f"Using slice thickness of {slice_thickness:.2e} m (3x mean pore diameter)")

# 3. Find the pores within the slices at the domain boundaries
min_coord = coords[:, axis].min()
max_coord = coords[:, axis].max()

inlet = np.where(coords[:, axis] < min_coord + slice_thickness)[0]
outlet = np.where(coords[:, axis] > max_coord - slice_thickness)[0]

# 4. Apply the boundary conditions
C_in, C_out = 1, 0

# Sanity check before applying
if inlet.size == 0 or outlet.size == 0:
    raise RuntimeError("Boundary condition setup failed: No pores found in inlet/outlet slices. Check slice_thickness or network geometry.")

print(f"Found {inlet.size} inlet pores and {outlet.size} outlet pores.")

fd.set_value_BC(pores=inlet, values=C_in)
fd.set_value_BC(pores=outlet, values=C_out)



print("Applied boundary conditions: Concentration = 1.0 at inlet, 0.0 at outlet.")
# --- APPLY BOUNDARY CONDITIONS ---


# --- RUN THE SIMULATION ---
print("\nRunning the solver... this may take a moment.")
fd.run()
print("Solver finished running.")


# --------------------------------------------------------------------------- #
# --- 3. POST-PROCESSING AND VISUALIZATION ---
# --------------------------------------------------------------------------- #
print("\n--- 3. POST-PROCESSING AND VISUALIZATION ---")

# --- CALCULATE EFFECTIVE DIFFUSIVITY ---
# This is the main result: a bulk property of the entire network.
rate_inlet = fd.rate(pores=inlet).sum()
print(f'Molar flow rate: {rate_inlet:.5e} mol/s')

# Always compute geometry from pore coordinates
coords = pn['pore.coords']
xmin, ymin, zmin = coords.min(axis=0)
xmax, ymax, zmax = coords.max(axis=0)

# Flow direction = x-axis
L = float(xmax - xmin)
A = float((ymax - ymin) * (zmax - zmin))

if A <= 0 or L <= 0:
    raise ValueError("Degenerate geometry: cross-section or length is zero.")

D_eff = rate_inlet * L / (A * (C_in - C_out))
print("Effective diffusivity: {0:.6E} m2/s".format(D_eff))

# Porosity from actual pore + throat volumes vs convex hull of coords
V_p = pn['pore.volume'].sum()
V_t = pn['throat.volume'].sum()
V_bulk = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
e = (V_p + V_t) / V_bulk
print('The porosity is: ', "{0:.6E}".format(e))

# Tortuosity: tau = e*D_AB / D_eff
D_AB = float(air['pore.diffusivity'][0]) if 'pore.diffusivity' in air.keys() else 1e-5
tau = e * D_AB / D_eff if D_eff != 0 else float('inf')
print('The tortuosity is:', "{0:.6E}".format(tau))

# ---------------- ML DESCRIPTORS (compute and optionally write to CSV) ----------------
try:
    descriptors = {}
    Np = int(pn.Np)
    Nt = int(pn.Nt)
    descriptors['Np'] = Np
    descriptors['Nt'] = Nt

    # Basic porosity (from volumes)
    descriptors['porosity_net'] = float(e)

    # Pore size stats (use pore.diameter if available, fallback to sphere-equivalent from volume)
    try:
        pdia = np.asarray(pn['pore.diameter']).astype(float)
    except Exception:
        try:
            pvol = np.asarray(pn['pore.volume']).astype(float)
            pdia = (6.0 * pvol / np.pi) ** (1.0 / 3.0)
        except Exception:
            pdia = np.full(Np, np.nan)
    descriptors['pore_diam_mean'] = float(np.nanmean(pdia))
    descriptors['pore_diam_var'] = float(np.nanvar(pdia))
    try:
        descriptors['pore_diam_skew'] = float(spst.skew(pdia[~np.isnan(pdia)]))
        descriptors['pore_diam_kurtosis'] = float(spst.kurtosis(pdia[~np.isnan(pdia)]))
    except Exception:
        descriptors['pore_diam_skew'] = float('nan')
        descriptors['pore_diam_kurtosis'] = float('nan')

    # Throat size stats
    try:
        tdia = np.asarray(pn['throat.diameter']).astype(float)
    except Exception:
        try:
            tdia = np.asarray(pn['throat.diameter_seed']).astype(float)
        except Exception:
            tdia = np.full(Nt, np.nan)
    descriptors['throat_diam_mean'] = float(np.nanmean(tdia))
    descriptors['throat_diam_var'] = float(np.nanvar(tdia))

    # Coordination number stats
    try:
        neigh = np.asarray(pn.num_neighbors(pn.Ps))
    except Exception:
        try:
            # fallback: build degree from throat.conns
            conns = np.asarray(pn['throat.conns'])
            neigh = np.bincount(conns.ravel(), minlength=Np)
        except Exception:
            neigh = np.full(Np, np.nan)
    descriptors['coord_mean'] = float(np.nanmean(neigh))
    descriptors['coord_var'] = float(np.nanvar(neigh))
    descriptors['fraction_dangling'] = float(np.count_nonzero(neigh == 1) / max(1, Np))

    # Connectivity / cluster features
    try:
        pore_clusters = op.topotools.find_clusters(pn)
        unique_ids, counts = np.unique(pore_clusters, return_counts=True)
        n_clusters = unique_ids.size
        n_isolated_clusters = int(np.sum(counts == 1))
        descriptors['n_clusters'] = int(n_clusters)
        descriptors['frac_isolated_clusters'] = float(n_isolated_clusters / max(1, n_clusters))
    except Exception:
        descriptors['n_clusters'] = -1
        descriptors['frac_isolated_clusters'] = float('nan')

    # Geodesic (shortest-path) stats: compute shortest distance from each inlet to nearest outlet
    try:
        coords = np.asarray(pn['pore.coords'])
        conns = np.asarray(pn['throat.conns'])
        # edge weights = Euclidean distances between pore centers
        p0 = coords[conns[:, 0]]
        p1 = coords[conns[:, 1]]
        lengths = np.linalg.norm(p0 - p1, axis=1)
        # build symmetric sparse adjacency
        rows = np.concatenate([conns[:, 0], conns[:, 1]])
        cols = np.concatenate([conns[:, 1], conns[:, 0]])
        data = np.concatenate([lengths, lengths])
        G = csr_matrix((data, (rows, cols)), shape=(Np, Np))
        # compute dijkstra distances from inlets
        inlet_idx = np.atleast_1d(inlet)
        outlet_idx = np.atleast_1d(outlet)
        if inlet_idx.size > 0 and outlet_idx.size > 0:
            dist_matrix = dijkstra(csgraph=G, directed=False, indices=inlet_idx)
            # dist_matrix shape (len(inlet), Np) or (N,) if single inlet
            # ensure 2D
            if dist_matrix.ndim == 1:
                dist_matrix = dist_matrix[np.newaxis, :]
            # for each inlet get min distance to any outlet
            min_dists = np.min(dist_matrix[:, outlet_idx], axis=1)
            descriptors['geodesic_mean'] = float(np.mean(min_dists))
            descriptors['geodesic_median'] = float(np.median(min_dists))
            descriptors['geodesic_std'] = float(np.std(min_dists))
        else:
            descriptors['geodesic_mean'] = float('nan')
            descriptors['geodesic_median'] = float('nan')
            descriptors['geodesic_std'] = float('nan')
    except Exception:
        descriptors['geodesic_mean'] = float('nan')
        descriptors['geodesic_median'] = float('nan')
        descriptors['geodesic_std'] = float('nan')

    # Accessible porosity: fraction of pore volume belonging to clusters that touch both inlet & outlet
    try:
        if 'pore.volume' in pn.keys():
            pvol = np.asarray(pn['pore.volume']).astype(float)
        else:
            pvol = np.ones(Np)
        if 'pore_clusters' not in locals():
            pore_clusters = op.topotools.find_clusters(pn)
        inlet_clusters = set(pore_clusters[inlet]) if inlet.size > 0 else set()
        outlet_clusters = set(pore_clusters[outlet]) if outlet.size > 0 else set()
        through_clusters = inlet_clusters.intersection(outlet_clusters)
        if len(through_clusters) > 0:
            mask_through = np.isin(pore_clusters, list(through_clusters))
            accessible_pore_vol = float(np.sum(pvol[mask_through]))
        else:
            accessible_pore_vol = 0.0
        descriptors['accessible_porosity_fraction'] = float(accessible_pore_vol / max(1e-30, np.sum(pvol)))
    except Exception:
        descriptors['accessible_porosity_fraction'] = float('nan')

    # Tortuosity: try OpenPNM metric, fallback to e*D_AB/D_eff
    try:
        # Try common signatures
        try:
            tort = op.metrics.tortuosity(pn, phase=air)
        except Exception:
            try:
                tort = op.metrics.tortuosity(pn)
            except Exception:
                tort = float('nan')
        descriptors['tortuosity_metric'] = float(tort) if np.isfinite(tort) else float('nan')
    except Exception:
        descriptors['tortuosity_metric'] = float('nan')

    # Path-based tortuosity from mean geodesic distance normalized by straight length
    try:
        if not np.isnan(descriptors['geodesic_mean']):
            straight_L = L
            descriptors['path_tortuosity'] = float(descriptors['geodesic_mean'] / straight_L) if straight_L > 0 else float('nan')
        else:
            descriptors['path_tortuosity'] = float('nan')
    except Exception:
        descriptors['path_tortuosity'] = float('nan')

    # Transport fields already computed
    descriptors['molar_flow_rate'] = float(rate_inlet)
    descriptors['D_eff'] = float(D_eff)
    descriptors['D_AB'] = float(D_AB)

    # Merge descriptors into a flat row and write to CSV unless tort==inf and D_eff==0
    do_skip = (not np.isfinite(tau) or np.isinf(tau)) and (D_eff == 0)
    if do_skip:
        print('Skipping CSV export: tortuosity is infinite and D_eff == 0')
    else:
        out_csv = 'ml_dataset.csv'
        row = {**descriptors}
        # write/appending
        df_row = pd.DataFrame([row])
        write_header = not os.path.exists(out_csv)
        df_row.to_csv(out_csv, mode='a', header=write_header, index=False)
        print(f'Appended descriptors to {out_csv}')
except Exception as exc:
    print('Could not compute all ML descriptors:', exc)
# -----------------------------------------------------------------------------------------

