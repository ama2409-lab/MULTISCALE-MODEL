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
from NETWORK_GENERATION_FINAL import build_voronoi_network

# Set a consistent style for the plots
op.visualization.set_mpl_style()

# =========================================================================
# --- 1. DEFINE THE PARAMETERS FOR THE UPSCALING & ML STUDY ---
# =========================================================================

# --- Domain sizes (L) to test, in meters ---
domain_lengths = np.array([100e-6, 150e-6, 200e-6, 250e-6, 300e-6, 350e-6])

# --- Random seeds to run for each domain size for statistical averaging ---
seeds = [42, 101, 211, 314, 555]

# --- Control Variable: Constant Pore Density (pores per cubic meter) ---
# Base this on a reference case (e.g., 500 points in a 250um cube)
base_points = 500
base_volume = (250e-6)**3
points_per_volume = base_points / base_volume
print(f"Targeting a constant pore density of {points_per_volume:.2e} pores/m^3")

# --- File to store all results ---
output_csv_path = 'upscaling_ml_dataset.csv'

# =========================================================================
# --- 2. THE MAIN SIMULATION LOOP ---
# =========================================================================
all_results = []

for L_domain in domain_lengths:
    domain_size = (L_domain, L_domain, L_domain)
    # Calculate the number of points needed to maintain constant density
    num_points = int(points_per_volume * (L_domain**3))

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"--- Running: L = {L_domain*1e6:.0f} um | {num_points} pores | seed = {seed} ---")
        print(f"{'='*60}")
        
        # Dictionary to store all data for this specific run
        run_data = {'domain_size_L': L_domain, 'num_points': num_points, 'seed': seed}
        
        try:
            # --- A. NETWORK GENERATION ---
            pn = build_voronoi_network(points=num_points, domain_size=domain_size,
                                       seed=seed) # save_png=False for batch runs
            
            # --- B. SIMULATION SETUP ---
            air = pn.project.phases()['Air']
            fd = op.algorithms.FickianDiffusion(network=pn, phase=air)

            # --- C. BOUNDARY CONDITIONS (The Correct Way) ---
            axis = 0  # Flow along X-axis
            coords = pn['pore.coords']
            mean_pore_dia = pn['pore.diameter'].mean()
            slice_thickness = 3 * mean_pore_dia
            
            min_coord = coords[:, axis].min()
            max_coord = coords[:, axis].max()
            
            inlet = np.where(coords[:, axis] < min_coord + slice_thickness)[0]
            outlet = np.where(coords[:, axis] > max_coord - slice_thickness)[0]
            
            if inlet.size == 0 or outlet.size == 0:
                print("⚠️ Warning: No boundary pores found. Skipping this run.")
                continue
            
            C_in, C_out = 1.0, 0.0
            fd.set_value_BC(pores=inlet, values=C_in)
            fd.set_value_BC(pores=outlet, values=C_out)

            # --- D. RUN SIMULATION ---
            fd.run()
            
            # --- E. POST-PROCESSING & ML DESCRIPTOR CALCULATION ---
            rate_inlet = fd.rate(pores=inlet)[0]
            
            # Use the defined domain size for consistency, not min/max of pore coords
            A = L_domain * L_domain
            D_eff = rate_inlet * L_domain / (A * (C_in - C_out)) if A > 0 else 0.0
            print(f"--> Effective Diffusivity (D_eff): {D_eff:.4e} m^2/s")
            
            # Add results to our dictionary
            run_data['D_eff'] = D_eff
            run_data['molar_flow_rate'] = rate_inlet

            V_p = pn['pore.volume'].sum()
            V_t = pn['throat.volume'].sum()
            porosity = (V_p + V_t) / (L_domain**3)
            run_data['porosity'] = porosity

            D_AB = air['pore.diffusivity'][0]
            tortuosity = (porosity * D_AB / D_eff) if D_eff > 0 else np.inf
            run_data['tortuosity'] = tortuosity

            # Add other geometric descriptors for ML
            pdia = pn['pore.diameter']
            run_data['pore_diam_mean'] = pdia.mean()
            run_data['pore_diam_std'] = pdia.std()
            
            tdia = pn['throat.diameter']
            run_data['throat_diam_mean'] = tdia.mean()
            run_data['throat_diam_std'] = tdia.std()
            
            neigh = pn.num_neighbors(pn.Ps)
            run_data['coord_mean'] = neigh.mean()
            run_data['coord_std'] = neigh.std()

            # Geodesic (shortest-path) stats
            conns = pn['throat.conns']
            lengths = np.linalg.norm(pn['pore.coords'][conns[:,0]] - pn['pore.coords'][conns[:,1]], axis=1)
            G = csr_matrix((lengths, (conns[:, 0], conns[:, 1])), shape=(pn.Np, pn.Np))
            dist_matrix = dijkstra(csgraph=G, directed=False, indices=inlet)
            min_dists = np.min(dist_matrix[:, outlet], axis=1)
            run_data['geodesic_mean'] = np.mean(min_dists[np.isfinite(min_dists)])
            
            # --- F. STORE THIS RUN'S DATA ---
            if D_eff == 0 or not np.isfinite(D_eff):
                print(f"Skipping storing run (L={L_domain*1e6:.0f} um, seed={seed}) because D_eff == 0 or not finite.")
            else:
                all_results.append(run_data)

        except Exception as e:
            print(f"❌ ERROR during simulation for L={L_domain*1e6:.0f} um, seed={seed}. Skipping.")
            print(f"   Error message: {e}")
            # Optionally log the error to a file
            with open("error_log.txt", "a") as f:
                f.write(f"L={L_domain}, seed={seed}, error: {e}\n")


# =========================================================================
# --- 3. SAVE & ANALYZE RESULTS ---
# =========================================================================
if not all_results:
    print("\nNo results were generated. Exiting.")
else:
    # --- Create a single DataFrame and save to CSV ---
    df = pd.DataFrame(all_results)
    # Reorder columns for clarity
    cols_order = ['domain_size_L', 'seed', 'num_points', 'D_eff', 'tortuosity', 'porosity', 
                  'pore_diam_mean', 'pore_diam_std', 'throat_diam_mean', 'throat_diam_std',
                  'coord_mean', 'coord_std', 'geodesic_mean', 'molar_flow_rate']
    df = df[[c for c in cols_order if c in df.columns]]
    df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Successfully saved all {len(df)} simulation results to '{output_csv_path}'")

    # --- Part 1: Upscaling Analysis (REV Plot) ---
    print("\n--- Generating REV Analysis Plot ---")
    rev_analysis = df.groupby('domain_size_L')['D_eff'].agg(['mean', 'std']).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        rev_analysis['domain_size_L'] * 1e6,  # x-axis in microns
        rev_analysis['mean'],
        yerr=rev_analysis['std'],
        fmt='-o',
        capsize=5,
        label='Effective Diffusivity'
    )
    ax.set_xlabel("Domain Size L (microns)")
    ax.set_ylabel("Effective Diffusivity D_eff (m^2/s)")
    ax.set_title("REV Analysis: D_eff vs. Domain Size")
    ax.grid(True, which='both', linestyle='--')
    ax.legend()
    plt.tight_layout()
    rev_plot_path = 'REV_analysis_plot.png'
    fig.savefig(rev_plot_path)
    print(f"Saved REV plot to '{rev_plot_path}'")

    # --- Part 2: Machine Learning Next Steps ---
    print("\n--- Machine Learning Dataset is Ready ---")
    print(f"Your dataset ('{output_csv_path}') has {df.shape[0]} rows and {df.shape[1]} columns.")
    print("\nTo start with ML, you can now:")
    print(f"1. Load the data: df = pd.read_csv('{output_csv_path}')")
    print("2. Define features (X) and target (y):")
    print("   features = ['porosity', 'pore_diam_mean', 'coord_mean', ...]")
    print("   X = df[features]")
    print("   y = df['D_eff']")
    print("3. Split data and train a model (e.g., RandomForestRegressor or XGBoost).")