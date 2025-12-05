#WORKS !!!
import openpnm as op
import numpy as np
import scipy.stats as spst
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

op.visualization.set_mpl_style()

print("--- NETWORKgeneration module loaded ---")


def build_voronoi_network(points=500, domain_size=(250e-6, 250e-6, 250e-6), seed=42, save_png=True):
    """Builds and returns a Voronoi OpenPNM network with sampled diameters and geometry.

    Parameters
    ----------
    points : int
        Number of Voronoi seed points (roughly the number of pores).
    domain_size : tuples
        Physical domain size in meters for the Voronoi generator.
    seed : int
        RNG seed used for reproducible sampling.
    save_png : bool
        Whether to save a PNG visualization of the network.

    Returns
    -------
    pn : openpnm.network.Network
        The prepared OpenPNM network with geometry and physics models.
    """
    # seed RNG for reproducibility
    rng = np.random.default_rng(int(seed))
    print("Step 1: Generating a Voronoi network.")

    # If 'points' is an integer, explicitly generate that many random 3D seed coordinates
    # inside the provided physical domain. This ensures different point counts produce
    # different tessellations rather than relying on ambiguous API behavior.
    points_coords = points
    if isinstance(points, (int, np.integer)):
        domain = np.asarray(domain_size, dtype=float)
        # generate uniformly inside [0, domain] for each axis
        points_coords = rng.random((int(points), 3)) * domain[np.newaxis, :]

    # Try preferred signature: pass explicit point coordinates to Voronoi
    try:
        pn = op.network.Voronoi(points=points_coords, shape=domain_size)
    except Exception:
        # Fall back to older signature (some OpenPNM versions accept integer points)
        pn = op.network.Voronoi(shape=domain_size, points=points)

    print(f"Generated Voronoi network with {pn.Np} pores and {pn.Nt} throats.")
    try:
        print(f"Average coordination number: {np.mean(pn.num_neighbors(pn.Ps)):.2f}")
    except Exception:
        pass

    # --- 2. Add Realistic GEOMETRY by Direct Assignment and Manual Calculation ---
    print("\nStep 2: Adding statistical geometry by direct array assignment.")

    # --- Pore Diameters ---
    # Define the SciPy distribution and explicitly draw samples.
    pore_diam_dist = spst.norm(loc=20e-6, scale=5e-6)
    pore_d_samples = pore_diam_dist.rvs(size=pn.Np)
    #pore_d_samples = spst.truncnorm(...).rvs(random_state=rng) <- IMPLEMENT?

    # Directly assign the clipped array to the network object.
    pn['pore.diameter'] = np.clip(pore_d_samples, a_min=1e-9, a_max=None)
    print("Assigned pore diameters from a normal distribution.")

    # --- Throat Diameters ---
    # Sample from the throat diameter distribution.
    throat_diam_dist = spst.weibull_min(c=2.5, loc=2e-6, scale=8e-6)
    throat_d_samples = throat_diam_dist.rvs(size=pn.Nt)
    #Same random_state issue for throat_d_samples. Use random_state=rng or rng.weibull approach

    # Assign the sampled values as a temporary "seed".
    #throat_len = np.linalg.norm(coords[p1] - coords[p2], axis=1)
    #max_d = np.minimum(max_d, 0.9*throat_len) 

    pn['throat.diameter_seed'] = np.clip(throat_d_samples, a_min=1e-9, a_max=None)

    

    # --- CORRECTED PHYSICAL CONSTRAINT ---
    # Manually calculate the maximum possible throat diameter instead of using a model.
    print("Manually calculating physical constraints for throat sizes...")
    # Get the connection info and the pore diameters
    conns = pn['throat.conns']
    p_diams = pn['pore.diameter']
    # For each throat, find the diameters of the two pores it connects
    P1_diams = p_diams[conns[:, 0]]
    P2_diams = p_diams[conns[:, 1]]
    # The maximum possible diameter is the smaller of the two pore diameters
    max_d = np.minimum(P1_diams, P2_diams)
    pn['throat.max_diameter'] = max_d

    # The final throat diameter is the smaller of our random sample and the physical limit.
    pn['throat.diameter'] = np.minimum(pn['throat.diameter_seed'], pn['throat.max_diameter'])
    print("Assigned throat diameters, constrained by pore sizes.")

    # --- 3. Add the Rest of the Geometry and Physics ---
    print("\nStep 3: Calculating dependent geometric and physical properties.")
    # This collection will RESPECT the existing diameter arrays and not overwrite them.
    # It will use them to calculate area, volume, etc.
    pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
    pn.regenerate_models()

    air = op.phase.Air(network=pn)
    air.add_model_collection(op.models.collections.physics.basic)
    air.regenerate_models()
    print("Network is now ready for simulation.")
    

    # Optionally save a visualization for quick inspection
    if save_png:
        print("Saving visualization...")
        fig = plt.figure(figsize=[8, 8])
        ax = fig.add_subplot(projection='3d')
        op.visualization.plot_connections(pn, ax=ax, color='grey')
        try:
            pdia = pn['pore.diameter']
            sizes = 200.0 * (pdia / np.nanmax(pdia))
            op.visualization.plot_coordinates(pn, ax=ax, size_by=sizes, color_by=pdia)
        except Exception:
            op.visualization.plot_coordinates(pn, ax=ax)
        ax.set_title("Complex Voronoi Network with Distributed Pore Sizes")
        out_png = 'network_voronoi_complex.png'
        fig.savefig(out_png, dpi=200, bbox_inches='tight')
        print(f"Saved network visualization to {out_png}")

    return pn


if __name__ == '__main__':
    # If run as a script, build a network and save a visualization
    pn = build_voronoi_network(points=100, domain_size=(250e-6, 250e-6, 250e-6), seed=42, save_png=True)
    print("Built network from __main__")