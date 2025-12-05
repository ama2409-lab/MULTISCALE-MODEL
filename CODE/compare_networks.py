import os
import sys
import traceback
import numpy as np

# Ensure CODE is importable
CODE_DIR = os.path.join(os.getcwd(), 'CODE')
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import NETWORK_GENERATION_FINAL as orig

import openpnm as op
import scipy.stats as spst
import warnings

# User's candidate function (copied as-is for testing)
def user_build_voronoi_network(domain_size, pore_density=None, points=None,
                          pore_size_dist={'name': 'lognorm', 's': 0.25, 'scale': 20e-6},
                          throat_size_dist={'name': 'weibull_min', 'c': 2.5, 'scale': 8e-6},
                          correlate_pore_throat=False,
                          beta_params=(5.0, 2.0),
                          seed=42,
                          save_files=False):
    rng = np.random.default_rng(seed)

    # --- Step 1. Determine points and generate seeds ---
    if pore_density is not None:
        if points is not None:
            warnings.warn("Both 'pore_density' and 'points' were specified. 'pore_density' takes precedence.")
        domain_vol = float(np.prod(domain_size))
        num_points = int(max(10, round(pore_density * domain_vol)))
        pts_arg = rng.random((num_points, 3)) * np.asarray(domain_size)
    elif points is not None:
        if isinstance(points, (int, np.integer)):
            num_points = int(points)
            pts_arg = rng.random((num_points, 3)) * np.asarray(domain_size)
        else: # Assume it's an array of coordinates
            pts_arg = np.asarray(points, dtype=float)
            num_points = pts_arg.shape[0]
    else:
        raise ValueError("Either 'pore_density' or 'points' must be specified.")

    if num_points < 10:
        raise ValueError(f"Too few points ({num_points}) to generate a meaningful network.")

    pn = op.network.Voronoi(points=pts_arg, shape=domain_size)
    pn['meta.seed'] = int(seed) # Use 'meta' namespace for convention

    # --- Step 2. Assign pore diameters ---
    pore_params = pore_size_dist.copy()
    pore_dist_func = getattr(spst, pore_params.pop('name'))
    pore_d_samples = pore_dist_func(**pore_params).rvs(size=pn.Np, random_state=rng)
    pn['pore.diameter'] = np.clip(pore_d_samples, a_min=1e-9, a_max=None)

    # --- Step 3. Assign throat diameters (correlated or independent) ---
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

    # --- Step 4. Add geometry, apply constraints, and regenerate ---
    pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
    # Note: using selective regenerate calls as the user's code does
    try:
        pn.regenerate_models('pore.*')
    except Exception:
        # Some OpenPNM versions may not support selectors; propagate to let caller handle
        raise
    pn.regenerate_models('throat.length')

    max_d_from_pores = np.minimum(pn['pore.diameter'][pn['throat.conns'][:, 0]],
                                  pn['pore.diameter'][pn['throat.conns'][:, 1]])
    max_d_from_length = pn['throat.length']
    max_d = np.minimum(max_d_from_pores, max_d_from_length)

    pn['throat.diameter'] = np.minimum(throat_d_samples, max_d)
    pn['throat.diameter'] = np.clip(pn['throat.diameter'], a_min=1e-9, a_max=None)

    # Final regeneration of all dependent models (geometry and physics)
    air = op.phase.Air(network=pn)
    air.add_model_collection(op.models.collections.physics.basic)
    pn.regenerate_models()
    air.regenerate_models()

    # Final check: Use assert for logical invariants (fail-fast)
    assert not np.any(pn['throat.diameter'] > pn['throat.length']), \
        "Logical Error: Throat diameters exceed their length despite constraints."

    return pn


# Helper to summarize a network

def summarize(pn):
    return {
        'Np': int(pn.Np),
        'Nt': int(pn.Nt),
        'avg_coord': float(np.mean(pn.num_neighbors(pn.Ps))),
        'mean_pore_d': float(np.mean(pn['pore.diameter'])),
        'mean_throat_d': float(np.mean(pn['throat.diameter'])),
        'max_throat_over_length': float(np.max(pn['throat.diameter'] - pn['throat.length']))
    }


# Test parameters
DOMAIN = (250e-6, 250e-6, 250e-6)
PORE_DENSITY = 1e14
SEED = 42

print('Running original build_voronoi_network...')
try:
    pn_orig = orig.build_voronoi_network(points=None, domain_size=DOMAIN, pore_density=PORE_DENSITY,
                                         pore_size_dist=None, throat_size_dist=None,
                                         correlate_pore_throat=False, seed=SEED, save_files=False)
    s_orig = summarize(pn_orig)
    print('Original summary:', s_orig)
except Exception as e:
    print('Original function raised an exception:')
    traceback.print_exc()

print('\nRunning user candidate build...')
try:
    pn_user = user_build_voronoi_network(domain_size=DOMAIN, pore_density=PORE_DENSITY, points=None,
                                         pore_size_dist={'name':'lognorm','s':0.25,'scale':20e-6},
                                         throat_size_dist={'name':'weibull_min','c':2.5,'scale':8e-6},
                                         correlate_pore_throat=False, beta_params=(5.0,2.0),
                                         seed=SEED, save_files=False)
    s_user = summarize(pn_user)
    print('User summary:', s_user)
except Exception as e:
    print('User function raised an exception:')
    traceback.print_exc()

# Compare key metrics if both succeeded
if 's_orig' in locals() and 's_user' in locals():
    print('\nComparison:')
    for k in s_orig.keys():
        print(f"{k}: original={s_orig[k]}  user={s_user[k]}  diff={s_user[k]-s_orig[k]}")
