import openpnm as op
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import Counter

# Make sure this script can find your network generation script
from NETWORK_GENERATION_FINAL import build_voronoi_network

op.visualization.set_mpl_style()

# =============================================================================
# STEP 1: SETUP FUNCTION (Logic identical to both scripts)
# =============================================================================
def setup_simulation(domain_size_m, seed):
    """
    Generates the network, repairs geometry, and sets up physics.
    Returns the OpenPNM network and phase objects.
    """
    print(f"--- Setting up domain: {domain_size_m*1e6:.0f} um ---")
    domain = (domain_size_m, domain_size_m, domain_size_m)
    density = 1e13
    num_points = int(max(10, round(density * np.prod(domain))))
    
    pn = build_voronoi_network(domain_size=domain, pore_density=density, points=num_points,
                               pore_size_dist={'name': 'lognorm', 's': 0.25, 'scale': 20e-6},
                               throat_size_dist={'name': 'weibull_min', 'c': 2.5, 'scale': 8e-6},
                               correlate_pore_throat=False, beta_params=(5.0, 2.0),
                               seed=seed, save_files=False)
    
    # --- Paste all your geometry repair logic here ---
    print('Running conservative geometry repairs...')
    # Repair throat.diameter
    td = pn['throat.diameter']
    bad_td = ~np.isfinite(td) | (td <= 0)
    if np.any(bad_td):
        mean_pos_td = np.mean(td[td > 0]) if np.any(td > 0) else 1e-9
        td[bad_td] = max(mean_pos_td * 0.1, 1e-9)
    # Repair throat.length
    tl = pn['throat.length']
    bad_tl = ~np.isfinite(tl) | (tl <= 0)
    if np.any(bad_tl):
        mean_pos_tl = np.mean(tl[tl > 0]) if np.any(tl > 0) else 1e-9
        tl[bad_tl] = max(mean_pos_tl * 0.1, 1e-9)
    
    # --- Paste your physics and conductance logic here ---
    air = op.phase.Air(network=pn)
    air.add_model_collection(op.models.collections.physics.basic)
    air.regenerate_models()
    
    print('Calculating diffusive conductance...')
    pore_diff = air['pore.diffusivity']
    conns = pn['throat.conns']
    td = pn['throat.diameter']
    tl = pn['throat.length']
    D_mean = np.mean(np.vstack([pore_diff[conns[:, 0]], pore_diff[conns[:, 1]]]), axis=0)
    D_mean[~np.isfinite(D_mean)] = np.nanmean(D_mean)
    area = np.pi * (td / 2.0)**2
    g = D_mean * area / tl
    eps_g = 1e-12
    g = np.where(np.isfinite(g) & (g > 0), g, eps_g)
    air['throat.diffusive_conductance'] = g

    return pn, air, domain, eps_g

# =============================================================================
# STEP 2: DIRICHLET SIMULATION FUNCTION
# =============================================================================
def run_dirichlet_sim(pn, air, domain, eps_g):
    """
    Runs the Dirichlet simulation and returns the effective diffusivity.
    """
    print("Running Dirichlet simulation...")
    fd = op.algorithms.FickianDiffusion(network=pn, phase=air)
    fd.settings['solver_family'] = 'pypardiso'

    # --- Paste your Dirichlet BC and cluster logic here ---
    coords = pn['pore.coords']
    axis = 0
    slice_thickness = 3 * np.mean(pn['pore.diameter'])
    minc, maxc = coords[:, axis].min(), coords[:, axis].max()
    inlet_pores = np.where(coords[:, axis] < minc + slice_thickness)[0]
    outlet_pores = np.where(coords[:, axis] > maxc - slice_thickness)[0]
    if inlet_pores.size == 0 or outlet_pores.size == 0: return np.nan

    C_in, C_out = 1.0, 0.0
    fd.set_value_BC(pores=inlet_pores, values=C_in)
    fd.set_value_BC(pores=outlet_pores, values=C_out)

    # Cluster logic
    conductive_mask = air['throat.diffusive_conductance'] > eps_g
    conns = pn['throat.conns'][conductive_mask]
    G = csr_matrix((np.ones(conns.shape[0]), (conns[:, 0], conns[:, 1])), shape=(pn.Np, pn.Np))
    ncomp, p_labels = connected_components(G, directed=False, connection='weak')
    labels_at_inlet = p_labels[inlet_pores]
    if len(labels_at_inlet) > 0:
        main_label = Counter(labels_at_inlet).most_common(1)[0][0]
        pores_to_solve = np.where(p_labels == main_label)[0]
        inlet_in_cluster = np.intersect1d(inlet_pores, pores_to_solve)
        outlet_in_cluster = np.intersect1d(outlet_pores, pores_to_solve)
        if outlet_in_cluster.size > 0:
            fd.set_value_BC(pores=inlet_in_cluster, values=C_in, mode='overwrite')
            fd.set_value_BC(pores=outlet_in_cluster, values=C_out, mode='overwrite')
        else:
            inlet_in_cluster, outlet_in_cluster = inlet_pores, outlet_pores
    else:
        inlet_in_cluster, outlet_in_cluster = inlet_pores, outlet_pores
        
    # --- Paste your Solver and D_eff calculation logic here ---
    fd.run()
    rate_in = fd.rate(pores=inlet_in_cluster).sum()
    L, A = domain[0], domain[1] * domain[2]
    D_eff = rate_in * L / (A * (C_in - C_out)) if abs(C_in - C_out) > 0 else 0.0
    print(f'Dirichlet D_eff: {D_eff:.6E} m2/s')
    
    # Clean up memory
    fd.project.clear(True)
    
    return D_eff

# =============================================================================
# STEP 3: NEUMANN SIMULATION FUNCTION
# =============================================================================
def run_neumann_sim(pn, air, domain, eps_g):
    """
    Runs the Neumann simulation and returns the effective diffusivity.
    """
    print("Running Neumann simulation...")
    fd = op.algorithms.FickianDiffusion(network=pn, phase=air)
    fd.settings['solver_family'] = 'pypardiso'

    # --- Paste your Neumann BC and cluster logic here ---
    coords = pn['pore.coords']
    axis = 0
    slice_thickness = 3 * np.mean(pn['pore.diameter'])
    minc, maxc = coords[:, axis].min(), coords[:, axis].max()
    inlet_pores = np.where(coords[:, axis] < minc + slice_thickness)[0]
    outlet_pores = np.where(coords[:, axis] > maxc - slice_thickness)[0]
    if inlet_pores.size == 0 or outlet_pores.size == 0: return np.nan

    rate_in_imposed = 1e-12 
    C_out = 0.0
    fd.set_rate_BC(pores=inlet_pores, rates=rate_in_imposed)
    fd.set_value_BC(pores=outlet_pores, values=C_out)

    # Cluster logic
    conductive_mask = air['throat.diffusive_conductance'] > eps_g
    conns = pn['throat.conns'][conductive_mask]
    G = csr_matrix((np.ones(conns.shape[0]), (conns[:, 0], conns[:, 1])), shape=(pn.Np, pn.Np))
    ncomp, p_labels = connected_components(G, directed=False, connection='weak')
    labels_at_inlet = p_labels[inlet_pores]
    if len(labels_at_inlet) > 0:
        main_label = Counter(labels_at_inlet).most_common(1)[0][0]
        pores_to_solve = np.where(p_labels == main_label)[0]
        inlet_in_cluster = np.intersect1d(inlet_pores, pores_to_solve)
        outlet_in_cluster = np.intersect1d(outlet_pores, pores_to_solve)
        if outlet_in_cluster.size > 0:
            fd.set_rate_BC(pores=inlet_in_cluster, rates=rate_in_imposed, mode='overwrite')
            fd.set_value_BC(pores=outlet_in_cluster, values=C_out, mode='overwrite')
        else:
            inlet_in_cluster, outlet_in_cluster = inlet_pores, outlet_pores
    else:
        inlet_in_cluster, outlet_in_cluster = inlet_pores, outlet_pores
        
    # --- Paste your Solver and D_eff calculation logic here ---
    fd.run()
    conc = fd['pore.concentration']
    C_in_avg = np.mean(conc[inlet_in_cluster])
    L, A = domain[0], domain[1] * domain[2]
    D_eff = rate_in_imposed * L / (A * (C_in_avg - C_out)) if abs(C_in_avg - C_out) > 0 else 0.0
    print(f'Neumann D_eff: {D_eff:.6E} m2/s')

    # Clean up memory
    fd.project.clear(True)
    
    return D_eff

# =============================================================================
# STEP 4: MAIN EXECUTION LOOP
# =============================================================================
if __name__ == "__main__":
    
    # Define the domain sizes (in meters) you want to test
    domain_sizes_to_test = np.array([150e-6, 200e-6, 250e-6, 300e-6, 400e-6, 500e-6])
    
    # Use the same seed for all simulations for a fair comparison
    SEED = 42 
    
    # Lists to store results
    results_dirichlet = []
    results_neumann = []

    print("--- Starting REV Study ---")
    
    for size in domain_sizes_to_test:
        # 1. Setup the network and physics
        # We must use the *same* network for both sims
        pn, air, domain, eps_g = setup_simulation(domain_size_m=size, seed=SEED)
        
        # 2. Run Dirichlet (Upper Bound)
        # Create a *copy* of the project to run the sim on
        proj_dirichlet = pn.project.copy()
        D_eff_D = run_dirichlet_sim(proj_dirichlet['net_01'], proj_dirichlet['phase_01'], domain, eps_g)
        results_dirichlet.append(D_eff_D)
        
        # 3. Run Neumann (Lower Bound)
        # Create another *copy* from the original
        proj_neumann = pn.project.copy()
        D_eff_N = run_neumann_sim(proj_neumann['net_01'], proj_neumann['phase_01'], domain, eps_g)
        results_neumann.append(D_eff_N)
        
        # 4. Clean up original project
        pn.project.clear(True)
        plt.close('all') # Close any open figures to save memory

    print("--- REV Study Complete ---")

    # =========================================================================
    # STEP 5: PLOT THE RESULTS
    # =========================================================================
    fig, ax = plt.subplots(figsize=[10, 6])
    
    # Plot domain size in microns for readability
    domain_sizes_um = domain_sizes_to_test * 1e6
    
    ax.plot(domain_sizes_um, results_dirichlet, 'bo-', label='Dirichlet (Upper Bound)')
    ax.plot(domain_sizes_um, results_neumann, 'ro-', label='Neumann (Lower Bound)')
    
    ax.set_xlabel('Domain Size (microns)')
    ax.set_ylabel('Effective Diffusivity (m$^2$/s)')
    ax.set_title('REV Convergence Study')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    fig.tight_layout()
    fig.savefig('rev_convergence_plot.png', dpi=300)
    plt.close(fig)

    print("Saved REV convergence plot to 'rev_convergence_plot.png'")