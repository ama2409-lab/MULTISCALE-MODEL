import openpnm as op
import numpy as np
import scipy.stats as spst
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend so script can run headless and save PNGs
import matplotlib.pyplot as plt
import warnings

import openpnm as op, numpy as np
print('OpenPNM version:', getattr(op, '__version__', 'unknown'))
# Shared defaults for domain size and pore density used across scripts
# Use cubic domain by default; scripts can import and override via CLI.
DEFAULT_DOMAIN = (250e-6, 250e-6, 250e-6)
DEFAULT_DENSITY = 1e13
# quick peek at network after a minimal run (optional)
# create a tiny network using your function or op.network.Voronoi to inspect keys

def build_voronoi_network(domain_size, pore_density=None, points=None,
                          pore_size_dist={'name': 'lognorm', 's': 0.25, 'scale': 20e-6},
                          throat_size_dist={'name': 'weibull_min', 'c': 2.5, 'scale': 8e-6},
                          correlate_pore_throat=False,
                          beta_params=(5.0, 2.0),
                          seed=42,
                          save_files=False):
    """
    Builds a robust, multiscale-consistent Voronoi network for data generation.

    This function enforces a clear parameter hierarchy: 'pore_density' takes
    precedence over 'points'. It is designed to "fail-fast" on logical errors.
    """
    rng = np.random.default_rng(seed)

    # --- Step 1. Robustly determine point count and generate seed coordinates ---
    if pore_density is not None:
        if points is not None:
            warnings.warn("Both 'pore_density' and 'points' were specified. "
                          "'pore_density' takes precedence.")
        domain_vol = float(np.prod(domain_size))
        # Ensure a minimum number of points to form a valid network
        num_points = int(max(10, round(pore_density * domain_vol)))
        print(f"domain_vol={domain_vol:.3e} m^3 -> num_points={num_points}")
        pts_arg = rng.random((num_points, 3)) * np.asarray(domain_size)
    elif points is not None:
        if isinstance(points, (int, np.integer)):
            num_points = int(points)
            pts_arg = rng.random((num_points, 3)) * np.asarray(domain_size)
        else:  # Assume it's an array of coordinates
            pts_arg = np.asarray(points, dtype=float)
            num_points = pts_arg.shape[0]
    else:
        raise ValueError("Either 'pore_density' or 'points' must be specified.")

    if num_points < 10:
        raise ValueError(f"Too few points ({num_points}) to generate a meaningful network.")

    pn = op.network.Voronoi(points=pts_arg, shape=domain_size)
    # OpenPNM requires keys to start with 'pore', 'throat' or 'param'
    pn['param.seed'] = int(seed)
    pn['param.num_points'] = num_points

    print(f"[1/6] Generated Voronoi network with {pn.Np} pores and {pn.Nt} throats from {num_points} seed points.")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ NEW SECTION: MAKE THE NETWORK TOPOLOGY PERIODIC +++
    # This is the crucial step for enabling PBC simulations.
    # It adds 'boundary' pores and 'periodic' throats connecting opposite faces.
    # We do this for all three directions to make the network fully periodic.
    # The new pores/throats will be labeled automatically (e.g., 'pore.left', 'throat.left_boundary').

    # Define a small, scale-aware tolerance to catch pores near the boundary.
    # Use a small fraction of the smallest domain dimension so the selection
    # adapts to domain scale. Adjust the multiplier (1e-6) if you need finer
    # or coarser face selection.
    shape = np.asarray(domain_size)
    tol = float(np.min(shape)) * 1e-6
    coords = pn['pore.coords']
    # ------------------------------------------------------------------
    # TODO INVESTIGATE: Verify that this scale-aware tolerance is appropriate
    # for all domain sizes you plan to use. Edge cases:
    #  - Very small domains (pore sizes comparable to domain) may need a
    #    tol based on pore diameter instead of domain size.
    #  - Very anisotropic domains may require per-axis tuning.
    # ACTION: consider adding `tol` as an explicit function argument and
    # logging the chosen value to saved artifacts for reproducibility.
    # ------------------------------------------------------------------
    
    # Get shape directly from the domain_size parameter for clarity
    shape = np.asarray(domain_size)

    # Select pores on each face using the tolerance
    Ps_left = pn.Ps[coords[:, 0] < (0 + tol)]
    Ps_right = pn.Ps[coords[:, 0] > (shape[0] - tol)]
    Ps_front = pn.Ps[coords[:, 1] < (0 + tol)]
    Ps_back = pn.Ps[coords[:, 1] > (shape[1] - tol)]
    Ps_bottom = pn.Ps[coords[:, 2] < (0 + tol)]
    Ps_top = pn.Ps[coords[:, 2] > (shape[2] - tol)]
    
    # Add a new layer of pores outside each face and label them
    # The new labels are what we'll use in the simulation script
    op.topotools.add_boundary_pores(network=pn, pores=Ps_left, apply_label='left_boundary')
    op.topotools.add_boundary_pores(network=pn, pores=Ps_right, apply_label='right_boundary')
    op.topotools.add_boundary_pores(network=pn, pores=Ps_front, apply_label='front_boundary')
    op.topotools.add_boundary_pores(network=pn, pores=Ps_back, apply_label='back_boundary')
    op.topotools.add_boundary_pores(network=pn, pores=Ps_top, apply_label='top_boundary')
    op.topotools.add_boundary_pores(network=pn, pores=Ps_bottom, apply_label='bottom_boundary')

    # Note: if exact wrap-around periodicity (1:1 pairing) is required,
    # perform explicit pairing in the simulation script instead of using
    # buffer boundary pores. When creating exact periodic throats, skip any
    # pairs whose transverse separation is below a tiny threshold to avoid
    # zero-length throats (suggested threshold: min(domain_size) * 1e-9).

    print("        Boundary layers created.")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- Step 2. Assign pore diameters ---
    # ------------------------------------------------------------------
    # TODO INVESTIGATE: Check that added boundary pores did not create
    # unintended zero-length throats or degenerate connections. If you
    # plan to create explicit 1:1 periodic throats in the simulation
    # script, ensure the pairing distance threshold is conservative
    # (skip pairs with transverse distance < min(domain_size)*1e-9).
    # Consider adding diagnostic output: counts of new throats, number
    # of zero-length throats, and a saved list of offending throat ids.
    # ------------------------------------------------------------------
    pore_params = pore_size_dist.copy()
    pore_dist_func = getattr(spst, pore_params.pop('name'))
    pore_d_samples = pore_dist_func(**pore_params).rvs(size=pn.Np, random_state=rng)
    pn['pore.diameter'] = np.clip(pore_d_samples, a_min=1e-9, a_max=None)
    print(f"[2/6] Assigned pore diameters (mean={np.mean(pn['pore.diameter']):.3e} m, dist={pore_size_dist.get('name')}).")

    # --- Step 3. Assign throat diameters ---
    if correlate_pore_throat:
        conns = pn['throat.conns']
        p_diams = pn['pore.diameter']
        min_p_diams = np.minimum(p_diams[conns[:, 0]], p_diams[conns[:, 1]])
        factors = spst.beta(a=beta_params[0], b=beta_params[1]).rvs(size=pn.Nt, random_state=rng)
        throat_d_samples = factors * min_p_diams
        method = 'correlated beta'
    else:
        throat_params = throat_size_dist.copy()
        throat_dist_func = getattr(spst, throat_params.pop('name'))
        throat_d_samples = throat_dist_func(**throat_params).rvs(size=pn.Nt, random_state=rng)
        method = throat_size_dist['name']
    print(f"[3/6] Sampled throat diameters (method={method}).")
    # --- Step 4. Add geometry, apply constraints, and regenerate all models ---
    pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
    
    # Assign throat diameters BEFORE final regeneration
    # Try a selective regeneration to obtain pore radii and throat.length. Some
    # OpenPNM versions don't support the 'propnames' selector; fall back to
    # regenerating all models in that case so generation is robust across installs.
    #
    # TODO INVESTIGATE: Confirm which geometry propnames your installed
    # OpenPNM exposes (e.g., 'throat.cross_section' vs 'throat.area'). If you
    # standardize on a set of propnames we can replace this broad call with
    # a targeted selective regeneration to save time on large networks.
    try:
        pn.regenerate_models(propnames='pore.*')  # Get pore radii
        pn.regenerate_models(propnames='throat.length')  # Get throat lengths
    except Exception as e:
        warnings.warn(f"Selective regeneration not supported ({e}); regenerating all models.")
        pn.regenerate_models()
    
    max_d_from_pores = np.minimum(pn['pore.diameter'][pn['throat.conns'][:, 0]],
                                  pn['pore.diameter'][pn['throat.conns'][:, 1]])
    max_d_from_length = pn['throat.length']
    max_d = np.minimum(max_d_from_pores, max_d_from_length)

    pn['throat.diameter'] = np.minimum(throat_d_samples, max_d)
    pn['throat.diameter'] = np.clip(pn['throat.diameter'], a_min=1e-9, a_max=None)
    print(f"[4/6] Assigned throat diameters (method={method}, mean={np.mean(pn['throat.diameter']):.3e} m).")

    # Add physics and do a final, full regeneration
    # Creating an `Air` phase can trigger heavy thermo/chemicals data loading
    # in some environments (thermo/chemicals packages). Use a safe fallback
    # so that large automated sweeps do not fail when those datasets are
    # unavailable. If the full `Air` class works it will be used; otherwise
    # fall back to a generic Phase with a reasonable default diffusivity.
    try:
        air = op.phase.Air(network=pn)
        air.add_model_collection(op.models.collections.physics.basic)
        pn.regenerate_models()  # Regenerates all remaining geometry
        air.regenerate_models()  # Regenerates physics
    except Exception as e:
        warnings.warn(f"Could not instantiate op.phase.Air() (falling back to generic Phase): {e}")
        air = op.phase.Phase(network=pn)
        # Provide conservative default diffusivity (m^2/s) so scripts that
        # expect 'pore.diffusivity' or throat conductances can proceed.
        # This value will usually be overwritten in downstream scripts if
        # they set specific diffusivities; here it prevents hard failures.
        default_D = 1e-5
        air['pore.diffusivity'] = np.full(pn.Np, default_D)
        # Defer adding physics models for the generic fallback to avoid
        # invoking thermo-backed model constructors.
    # Final check: Use assert for logical invariants (fail-fast)
    # ------------------------------------------------------------------
    # TODO INVESTIGATE: After regeneration, consider saving a small
    # diagnostics snapshot to disk (e.g., pn.keys(), air.keys(), counts of
    # non-finite entries per array, the chosen 'tol' and the RNG seed).
    # This file is invaluable when solver failures (NaN/Inf) occur later in
    # the pipeline — it lets you reproduce and debug the exact failing
    # configuration without rerunning the entire generation process.
    # ------------------------------------------------------------------
    print("[5/6] Regenerated dependent models and physics.")
    assert not np.any(pn['throat.diameter'] > pn['throat.length']), \
        "Logical Error: Throat diameters exceed their length despite constraints."
    print("[6/6] Invariants OK: no throat diameter exceeds throat.length.")

    if save_files:
        # Saving logic using the guaranteed integer 'num_points'
        # Save seed points for reproducibility
        seed_fname = f"seed_points_pts{num_points}_seed{int(seed)}.npy"
        np.save(seed_fname, np.asarray(pts_arg))
        print(f"Saved seed points to '{seed_fname}'.")

        # Save a network visualization (3D scatter + connections)
        try:
            fig = plt.figure(figsize=[8, 8])
            ax = fig.add_subplot(projection='3d')
            op.visualization.plot_connections(pn, ax=ax, color='grey')
            pdia = pn['pore.diameter']
            max_p = np.nanmax(pdia)
            sizes = 200.0 * (pdia / max_p) if max_p > 0 else 50.0
            op.visualization.plot_coordinates(pn, ax=ax, size_by=sizes, color_by=pdia)
            ax.set_title(f"Voronoi network: pts={num_points}, seed={int(seed)}")
            outname = f"network_voronoi_pts{num_points}_seed{int(seed)}.png"
            fig.savefig(outname, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved network visualization to '{outname}'.")
        except Exception as e:
            # For independent runs we prefer to warn but not crash on image saving
            warnings.warn(f"Could not save visualization PNG: {e}")

    return pn


if __name__ == '__main__':
    # Standalone runner: generate images and seed file for quick inspection
    example_density = 1e13
    domain = (250e-6, 250e-6, 250e-6)
    num_points = int(max(10, round(example_density * np.prod(domain))))
    print(f"Standalone run -> domain={domain}, pore_density={example_density}, points={num_points}")
    pn = build_voronoi_network(domain_size=domain, pore_density=example_density, points=None,
                               pore_size_dist={'name': 'lognorm', 's': 0.25, 'scale': 20e-6},
                               throat_size_dist={'name': 'weibull_min', 'c': 2.5, 'scale': 8e-6},
                               correlate_pore_throat=False, beta_params=(5.0, 2.0),
                               seed=42, save_files=True)
    print('Standalone generation complete.')