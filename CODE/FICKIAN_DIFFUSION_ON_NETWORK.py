
#trash
import openpnm as op
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Minimal, headless tutorial example: Effective diffusivity and tortuosity
# This follows the OpenPNM tutorial for a Cubic network and is deterministic.

op.visualization.set_mpl_style()

np.random.seed(10)

print("--- Simple Fickian Diffusion Tutorial (Cubic network) ---")

# 1) Create a simple cubic network
shape = [10, 10, 1]
spacing = 1e-5
net = op.network.Cubic(shape=shape, spacing=spacing)
print(f"Created Cubic network: pores={net.Np}, throats={net.Nt}")

# 2) Add geometry models (spheres and cylinders)
geo = op.models.collections.geometry.spheres_and_cylinders
net.add_model_collection(geo, domain='all')
net.regenerate_models()

# 3) Create an 'Air' phase and add basic physics
air = op.phase.Air(network=net)
phys = op.models.collections.physics.basic
if 'throat.entry_pressure' in phys:
    # remove entry_pressure model if present in the collection (not needed)
    try:
        del phys['throat.entry_pressure']
    except Exception:
        pass
air.add_model_collection(phys)
air.regenerate_models()

# 4) Set up and run Fickian diffusion with Dirichlet BCs on left/right faces
fd = op.algorithms.FickianDiffusion(network=net, phase=air)
inlet = net.pores('left')
outlet = net.pores('right')
if inlet.size == 0 or outlet.size == 0:
    raise RuntimeError('Could not find inlet/outlet pores on network faces')

C_in, C_out = 10.0, 5.0
fd.set_value_BC(pores=inlet, values=C_in)
fd.set_value_BC(pores=outlet, values=C_out)

print(f"Applying Dirichlet BCs: C_in={C_in} on {inlet.size} pores, C_out={C_out} on {outlet.size} pores")

fd.run()
print('Solver finished')

# 5) Post-process: compute molar flow rate, effective diffusivity and tortuosity
rate_inlet = fd.rate(pores=inlet).sum()
print(f'Molar flow rate: {rate_inlet:.5e} mol/s')

# Domain geometry
A = (shape[1] * shape[2]) * (spacing ** 2)
L = shape[0] * spacing

if A <= 0 or L <= 0:
    raise ValueError('Degenerate domain geometry')

D_eff = rate_inlet * L / (A * (C_in - C_out))
print('Effective diffusivity: {0:.6E} m2/s'.format(D_eff))

# Porosity from pore+throat volumes vs bulk
V_p = net['pore.volume'].sum()
V_t = net['throat.volume'].sum()
V_bulk = np.prod(shape) * (spacing ** 3)
porosity = (V_p + V_t) / V_bulk
print('Porosity: {0:.6E}'.format(porosity))

# Tortuosity = porosity * D_AB / D_eff
D_AB = float(air['pore.diffusivity'][0]) if 'pore.diffusivity' in air.keys() else 1e-5
tau = porosity * D_AB / D_eff if D_eff != 0 else float('inf')
print('Tortuosity: {0:.6E}'.format(tau))

# 6) Save a small visualization (headless)
fig, ax = plt.subplots(figsize=(5, 5))
pc = fd['pore.concentration']
op.visualization.plot_coordinates(network=net, color_by=pc, size_by=net['pore.diameter'], markersize=300, ax=ax)
op.visualization.plot_connections(network=net, color_by=fd.interpolate_data('throat.concentration'), linewidth=2, ax=ax)
ax.set_axis_off()
fig.tight_layout()
outpng = 'fickian_cubic_concentration.png'
fig.savefig(outpng, dpi=200)
plt.close(fig)
print(f'Saved concentration plot to {outpng}')

print('Done.')
import openpnm as op
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set a consistent style for the plots
op.visualization.set_mpl_style()

# --------------------------------------------------------------------------- #
# --- 1. PORE NETWORK GENERATION (Directly in OpenPNM) ---
# --------------------------------------------------------------------------- #
print("--- 1. PORE NETWORK GENERATION ---")
# Build a Voronoi-derived network using the helper in NETWORKgeneration.py
# This part is correct and remains unchanged.
from NETWORK_GENERATION_FINAL import build_voronoi_network

# Create the network (adjust points and seed as desired)
density = 1e13
domain = (250e-6, 250e-6, 250e-6)
num_points = int(max(10, round(density * np.prod(domain))))
pn = build_voronoi_network(domain_size=domain, pore_density=density, points=num_points,
                               pore_size_dist={'name': 'lognorm', 's': 0.25, 'scale': 20e-6},
                               throat_size_dist={'name': 'weibull_min', 'c': 2.5, 'scale': 8e-6},
                               correlate_pore_throat=False, beta_params=(5.0, 2.0),
                               seed=42, save_files=False
                               )
print(f"Using Voronoi network with {pn.Np} pores and {pn.Nt} throats.")

# Quick health check (non-fatal)
try:
    h = op.utils.check_network_health(pn)
except Exception:
    h = None

# --------------------------------------------------------------------------- #
# --- 2. DIFFUSION SIMULATION SETUP ---
# --------------------------------------------------------------------------- #
print("\n--- 2. DIFFUSION SIMULATION SETUP ---")

# --- CREATE PHASE AND PHYSICS ---
# This part is correct and remains unchanged.
print("Creating an 'Air' phase object...")
air = op.phase.Air(network=pn)

print("Adding 'basic' physics models to the Air phase (if missing)...")
phys = op.models.collections.physics.basic
air.add_model_collection(phys)
air.regenerate_models()
print("Physics models ready (transport properties available).")

# --- SETUP THE ALGORITHM ---
print("\nSetting up the 'FickianDiffusion' algorithm...")
fd = op.algorithms.FickianDiffusion(network=pn, phase=air)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ CHANGE 1: APPLY BOUNDARY CONDITIONS USING LABELS (NOT SLICING) +++
# This is the correct way to use the network you generated.
print("--- Setting up Boundary Conditions using Network Labels ---")

# axis of flow (0=X, 1=Y, 2=Z)
axis = 0
C_in, C_out = 1.0, 0.0

# Find pores using the labels created in the network generation script
inlet_pores = pn.pores('left_boundary')
outlet_pores = pn.pores('right_boundary')

# Sanity check
if inlet_pores.size == 0 or outlet_pores.size == 0:
    raise ValueError("Boundary pore sets are empty! Check network generation.")

# Apply the boundary conditions to these specific pores
fd.set_value_BC(pores=inlet_pores, values=C_in)
fd.set_value_BC(pores=outlet_pores, values=C_out)

print(f"Applied BCs: C={C_in} on {len(inlet_pores)} 'left_boundary' pores.")
print(f"Applied BCs: C={C_out} on {len(outlet_pores)} 'right_boundary' pores.")

# --- FIX 1: HANDLE NON-FINITE CONDUCTANCES (Essential Step) ---
print("\nScanning for and correcting any non-finite conductance values...")
g_d = air['throat.diffusive_conductance']
g_d[~np.isfinite(g_d)] = 0.0  # Find and replace all non-finite values in one line
air['throat.diffusive_conductance'] = g_d
print("Conductance array is now clean.")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ SIMPLIFIED & CORRECT FIX 2: SOLVE ON THE MAIN CONDUCTIVE CLUSTER +++

print("\nIdentifying the main conductive cluster...")

# 1. Define the "mask" for find_clusters. We will use bond percolation.
#    A throat is part of the cluster if its conductance is positive.
conductive_throats = air['throat.diffusive_conductance'] > 0

# 2. Instead of relying on op.topotools.find_clusters (which may behave
#    differently across OpenPNM versions), build a sparse adjacency for the
#    conductive throats and compute connected components with SciPy. This
#    guarantees we get integer labels for pores.
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import Counter

conns = pn['throat.conns'][conductive_throats]
if conns.size == 0:
    raise RuntimeError('No conductive throats found; cannot identify conductive cluster.')
rows = np.concatenate([conns[:, 0], conns[:, 1]])
cols = np.concatenate([conns[:, 1], conns[:, 0]])
data = np.ones(rows.size, dtype=int)
G = csr_matrix((data, (rows, cols)), shape=(pn.Np, pn.Np))
num_comp, p_labels = connected_components(G, directed=False, connection='weak')

# 3. Find the cluster label that is connected to our inlet pores. Choose the
#    most common label among the inlet pores (robust to multipart inlets).
labels_at_inlet = p_labels[inlet_pores]
cnt = Counter(labels_at_inlet)
if len(cnt) == 0:
    raise RuntimeError('No cluster labels found for inlet pores. Check conductive throat mask.')
main_cluster_label = int(cnt.most_common(1)[0][0])

# 4. Get all pores belonging to this main cluster.
pores_to_solve = np.where(p_labels == main_cluster_label)[0]

print(f"Main cluster identified (label #{main_cluster_label}), containing {len(pores_to_solve)} pores.")

# Restrict boundary conditions to pores that belong to the main conductive cluster
inlet_in_cluster = np.intersect1d(inlet_pores, pores_to_solve)
outlet_in_cluster = np.intersect1d(outlet_pores, pores_to_solve)
if inlet_in_cluster.size == 0 or outlet_in_cluster.size == 0:
    # If the conductive cluster does not include our boundary pores it likely
    # means the conductance mask is too aggressive (many zeros). Fall back to
    # applying BCs to the original labeled boundary pores and continue.
    print("Warning: conductive cluster does not include inlet/outlet pores; falling back to label-based BCs on original boundary pores.")
    inlet_in_cluster = inlet_pores
    outlet_in_cluster = outlet_pores

# Re-apply BCs limited to the conductive cluster (harmless if applied earlier)
fd.set_value_BC(pores=inlet_in_cluster, values=C_in, mode='overwrite')
fd.set_value_BC(pores=outlet_in_cluster, values=C_out, mode='overwrite')
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# --- RUN THE SIMULATION ---
print("\nRunning the solver... this may take a moment.")
# The solver in fd.run() is smart enough to ignore the NaN values
# when it constructs the A and b matrices.
# QUICK FIX: detect pores that have zero total incident conductance (isolated)
# and pin them to a Dirichlet value to avoid a singular matrix A.
g_d_all = air['throat.diffusive_conductance']
conns_all = pn['throat.conns']
incident_g = np.zeros(pn.Np, dtype=float)
for i, (a, b) in enumerate(conns_all):
    try:
        incident_g[a] += float(g_d_all[i])
        incident_g[b] += float(g_d_all[i])
    except Exception:
        incident_g[a] += float(np.asarray(g_d_all[i]))
        incident_g[b] += float(np.asarray(g_d_all[i]))
isolated_pores = np.where(incident_g == 0)[0]
if isolated_pores.size > 0:
    print(f"Found {isolated_pores.size} isolated pores (zero incident conductance). Pinning them to C_in to avoid singular matrix.")
    fd.set_value_BC(pores=isolated_pores, values=C_in, mode='overwrite')

fd.run()
print("Solver finished running.")


# --------------------------------------------------------------------------- #
# --- 3. POST-PROCESSING AND VISUALIZATION ---
# --------------------------------------------------------------------------- #
print("\n--- 3. POST-PROCESSING AND VISUALIZATION ---")

# --- CALCULATE EFFECTIVE DIFFUSIVITY ---
# The rate calculation now uses the correct 'inlet_pores' variable
rate_inlet = fd.rate(pores=inlet_in_cluster).sum()
print(f'Molar flow rate: {rate_inlet:.5e} mol/s')


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ CHANGE 2: USE DEFINED DOMAIN SIZE FOR GEOMETRY CALCULATIONS +++
# This is critical because the min/max of pore coordinates is no longer
# representative of the domain size after adding boundary layers.

# Flow direction = x-axis
L = domain[axis]
# Get the other two dimensions for the cross-sectional area
cross_dims = [i for i in [0, 1, 2] if i != axis]
A = domain[cross_dims[0]] * domain[cross_dims[1]]

if A <= 0 or L <= 0:
    raise ValueError("Degenerate geometry: cross-section or length is zero.")
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

D_eff = rate_inlet * L / (A * (C_in - C_out))
print("Effective diffusivity: {0:.6E} m2/s".format(D_eff))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ CHANGE 3: CORRECT POROSITY CALCULATION +++
# Calculate porosity using the total pore+throat volume vs the defined bulk volume
internal_pores = pn.pores('internal') # Only consider internal pores for porosity
internal_throats = pn.find_neighbor_throats(pores=internal_pores, mode='xnor')
V_p = pn['pore.volume'][internal_pores].sum()
V_t = pn['throat.volume'][internal_throats].sum()
V_bulk = np.prod(domain)
e = (V_p + V_t) / V_bulk
# (conductance array already cleaned above)


# Tortuosity: tau = e*D_AB / D_eff
D_AB = float(air['pore.diffusivity'][0]) if 'pore.diffusivity' in air.keys() else 1e-5
tau = e * D_AB / D_eff if D_eff != 0 else float('inf')
print('The tortuosity is:', "{0:.6E}".format(tau))

# --- GENERATE PLOTS TO SEE WHAT'S HAPPENING ---
# This section is correct and remains unchanged. It will now also visualize
# the concentration on the new boundary pores, which is very helpful.
print("\nGenerating plots to visualize the results...")
print("--> Generating 3D plot of the network with concentration profile...")
try:
    fig1 = plt.figure(figsize=[8, 8])
    ax1 = fig1.add_subplot(projection='3d')
    op.visualization.plot_connections(pn, ax=ax1, color='grey')
    conc = fd['pore.concentration']
    op.visualization.plot_coordinates(
        pn,
        ax=ax1,
        color_by=conc,
        size_by=pn['pore.diameter'],
        markersize=250
    )
    ax1.set_title("3D Network with Pore Concentration")
    print("    WHAT TO LOOK FOR: A clear color gradient across the network, with")
    print("    the boundary pores showing solid high (inlet) and low (outlet) concentrations.")
    fig1.tight_layout()
    out3d = 'diffusion_3d_concentration.png'
    fig1.savefig(out3d, dpi=200, bbox_inches='tight')
    plt.close(fig1) # Close the figure to free up memory
    print(f"Saved 3D concentration plot to {out3d}")
except Exception as e:
    print(f"Could not generate 3D plot. Error: {e}")