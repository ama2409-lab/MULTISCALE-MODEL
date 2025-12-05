import openpnm as op
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Simple, minimal example: Fickian diffusion with periodic BCs along X-axis
# Uses your NETWORK_GENERATION_FINAL.build_voronoi_network builder.

from NETWORK_GENERATION_FINAL import build_voronoi_network

# --- Parameters ---
domain = (250e-6, 250e-6, 250e-6)
seed = 42
pore_density = 1e13
axis = 0  # flow axis (0=x,1=y,2=z)
Q_total = 1.0  # source/sink rate (arbitrary units)

# --- 1) Build network ---
print('Building network...')
pn = build_voronoi_network(domain_size=domain, pore_density=pore_density, points=None,
                           seed=seed, save_files=False)
print(f'Network: Np={pn.Np}, Nt={pn.Nt}')

# Provide a default surface tension property to prevent repeated OpenPNM warnings
# about missing 'throat.surface_tension' when some physics/models attempt to run.
try:
    if 'throat.surface_tension' not in pn.keys():
        pn['throat.surface_tension'] = np.ones(pn.Nt) * 0.072  # N/m (typical water-air)
    if 'pore.surface_tension' not in pn.keys():
        pn['pore.surface_tension'] = np.ones(pn.Np) * 0.072
    # Also provide a default contact angle (radians) to avoid missing-property warnings
    if 'throat.contact_angle' not in pn.keys():
        pn['throat.contact_angle'] = np.zeros(pn.Nt)
    if 'pore.contact_angle' not in pn.keys():
        pn['pore.contact_angle'] = np.zeros(pn.Np)
except Exception:
    # If network doesn't support direct property set at this point, ignore silently
    pass

# --- 2) Find internal face pores (pre-boundary) and boundary pores created by builder ---
coords = pn['pore.coords']
shape = np.asarray(domain)
# Use same tol as builder
tol = 1e-9
face_left_internal = pn.Ps[coords[:, axis] < (0 + tol)]
face_right_internal = pn.Ps[coords[:, axis] > (shape[axis] - tol)]

left_boundary = pn.pores('left_boundary')
right_boundary = pn.pores('right_boundary')

print('Counts:', 'left_internal', len(face_left_internal), 'right_internal', len(face_right_internal),
      'left_boundary', len(left_boundary), 'right_boundary', len(right_boundary))

# Quick sanity prints to help diagnose zero ΔC issues
print('Sample face_left_internal indices:', face_left_internal[:10])
print('Sample face_right_internal indices:', face_right_internal[:10])
print('left_boundary sample:', left_boundary[:10])
print('right_boundary sample:', right_boundary[:10])
print('Intersection left_internal & left_boundary:', np.intersect1d(face_left_internal, left_boundary))
print('Intersection right_internal & right_boundary:', np.intersect1d(face_right_internal, right_boundary))

# --- 3) Pair internal face pores to boundary pores and create periodic throats ---
# We'll pair each internal pore on a face to the nearest boundary pore on the opposite face
# based on the two transverse coordinates (axes other than flow axis).
other_axes = tuple(i for i in (0, 1, 2) if i != axis)

# Helper to pair and connect
def pair_and_connect(internal_idxs, boundary_idxs, axes=other_axes, label='periodic'):
    if internal_idxs.size == 0 or boundary_idxs.size == 0:
        print('Warning: empty set for pairing; skipping connect for this face')
        return
    # Skip if throats with this label already exist
    try:
        existing = pn.throats(label)
    except Exception:
        existing = np.array([], dtype=int)
    if existing.size > 0:
        print(f"Skipping pairing for label={label}: {existing.size} throats already present")
        return
    # KDTree on boundary coordinates (transverse axes)
    b_coords = coords[boundary_idxs][:, axes]
    tree = cKDTree(b_coords)
    i_coords = coords[internal_idxs][:, axes]
    dists, idxs = tree.query(i_coords)
    paired_boundary = boundary_idxs[idxs]
    # Compute full 3D distances to avoid creating zero-length throats
    full_dists = np.linalg.norm(coords[internal_idxs] - coords[paired_boundary], axis=1)
    # Minimum acceptable throat length (scale-aware)
    min_sep = float(np.min(shape)) * 1e-12
    # Detect any self-pairs (internal index equals paired boundary index)
    try:
        self_pairs = np.where(internal_idxs == paired_boundary)[0]
        if self_pairs.size > 0:
            print(f'Warning: {self_pairs.size} self-pairs found (internal==boundary indices).')
    except Exception:
        # If shapes mismatch or types unexpected, skip
        pass
    mask = full_dists > min_sep
    if not np.any(mask):
        print(f'All candidate pairs for label={label} are below min_sep={min_sep}; skipping')
        return
    sel_internal = internal_idxs[mask]
    sel_boundary = paired_boundary[mask]
    # Connect only filtered pairs — do one-to-one connects to avoid accidental all-to-all
    existing_conns = np.asarray(pn['throat.conns'])
    connected = 0
    skipped_existing = 0
    for p1, p2 in zip(sel_internal, sel_boundary):
        # check if connection already exists (either orientation)
        exists = False
        if existing_conns.size > 0:
            # existing_conns is (Nt,2)
            exists = np.any(np.all(existing_conns == [p1, p2], axis=1) | np.all(existing_conns == [p2, p1], axis=1))
        if exists:
            skipped_existing += 1
            continue
        # create the throat for this pair
        op.topotools.connect_pores(network=pn, pores1=[p1], pores2=[p2], labels=[label])
        connected += 1
        # append to existing_conns for further checks
        try:
            existing_conns = np.vstack([existing_conns, [p1, p2]]) if existing_conns.size>0 else np.array([[p1, p2]])
        except Exception:
            existing_conns = np.array([[p1, p2]])

    print(f'Connected {connected} periodic throats (label={label}); skipped {np.count_nonzero(~mask)} near-zero pairs, {skipped_existing} existing')


# Small helper used by the extended per-axis routine.
# Returns the internal face pores on the given axis using the same tol/shape/coords.
def _face_internal_pores_for_axis(ax):
    left = pn.Ps[coords[:, ax] < (0 + tol)]
    right = pn.Ps[coords[:, ax] > (shape[ax] - tol)]
    return left, right

# Connect right internal -> left boundary, and left internal -> right boundary
pair_and_connect(face_right_internal, left_boundary, label='periodic')
pair_and_connect(face_left_internal, right_boundary, label='periodic')

# Inspect newly created periodic throats (if any) to spot zero-length or self-connections
try:
    per_idx = pn.throats('periodic')
    if per_idx.size > 0:
        conns = np.asarray(pn['throat.conns'])[per_idx]
        lengths = np.asarray(pn['throat.length'])[per_idx]
        # conductance might not exist yet on 'air' until phase created; guard access
        gvals = None
        try:
            gvals = np.asarray(pn.project.phases[0]['throat.diffusive_conductance']) if hasattr(pn, 'project') and len(pn.project.phases)>0 else None
        except Exception:
            gvals = None
        print(f'Periodic throats count: {per_idx.size}. Sample (idx, conns, length, conductance):')
        for ii, t in enumerate(per_idx[:10]):
            gv = gvals[ii] if (gvals is not None and ii < gvals.size) else 'n/a'
            print(t, conns[ii].tolist(), float(lengths[ii]), gv)
except Exception as _:
    print('Could not inspect periodic throats (may be before phase creation)')

# Regenerate models for new topology
air = op.phase.Air(network=pn)
air.add_model_collection(op.models.collections.physics.basic)
# Mirror surface_tension onto the phase so physics/models that look on the phase
# find the property (avoids warnings during model execution).
try:
    if 'throat.surface_tension' not in air.keys() and 'throat.surface_tension' in pn.keys():
        air['throat.surface_tension'] = np.asarray(pn['throat.surface_tension'])
    if 'pore.surface_tension' not in air.keys() and 'pore.surface_tension' in pn.keys():
        air['pore.surface_tension'] = np.asarray(pn['pore.surface_tension'])
    # Mirror contact_angle as well
    if 'throat.contact_angle' not in air.keys() and 'throat.contact_angle' in pn.keys():
        air['throat.contact_angle'] = np.asarray(pn['throat.contact_angle'])
    if 'pore.contact_angle' not in air.keys() and 'pore.contact_angle' in pn.keys():
        air['pore.contact_angle'] = np.asarray(pn['pore.contact_angle'])
    # Also set these on any existing physics objects attached to the network
    try:
        for phys in pn.physics():
            if 'throat.surface_tension' not in phys.keys() and 'throat.surface_tension' in pn.keys():
                phys['throat.surface_tension'] = np.asarray(pn['throat.surface_tension'])
            if 'pore.surface_tension' not in phys.keys() and 'pore.surface_tension' in pn.keys():
                phys['pore.surface_tension'] = np.asarray(pn['pore.surface_tension'])
            if 'throat.contact_angle' not in phys.keys() and 'throat.contact_angle' in pn.keys():
                phys['throat.contact_angle'] = np.asarray(pn['throat.contact_angle'])
            if 'pore.contact_angle' not in phys.keys() and 'pore.contact_angle' in pn.keys():
                phys['pore.contact_angle'] = np.asarray(pn['pore.contact_angle'])
    except Exception:
        pass
except Exception:
    pass
pn.regenerate_models()
air.regenerate_models()

# Repair any zero/non-finite throat diameters or non-positive throat lengths
# that are causing NaN conductances. Set conservative small positive fallbacks
# and regenerate models so conductances are recomputed.
try:
    td = np.asarray(pn['throat.diameter']) if 'throat.diameter' in pn.keys() else None
    tl = np.asarray(pn['throat.length']) if 'throat.length' in pn.keys() else None
    replaced = False
    if td is not None:
        bad_td = np.where((~np.isfinite(td)) | (td <= 0))[0]
        if bad_td.size > 0:
            # choose small_eps based on existing positive diameters if available
            pos = td[np.isfinite(td) & (td > 0)]
            if pos.size > 0:
                small_eps = float(np.min(pos) * 0.1)
            else:
                small_eps = 1e-9
            print(f'Fixing {bad_td.size} throat.diameter entries <=0 or non-finite -> {small_eps}')
            td[bad_td] = small_eps
            pn['throat.diameter'] = td
            replaced = True
    if tl is not None:
        bad_tl = np.where((~np.isfinite(tl)) | (tl <= 0))[0]
        if bad_tl.size > 0:
            small_len = float(np.min(shape) * 1e-6)
            print(f'Fixing {bad_tl.size} throat.length entries <=0 or non-finite -> {small_len}')
            tl[bad_tl] = small_len
            pn['throat.length'] = tl
            replaced = True
    if replaced:
        # Recompute dependent models after these repairs
        try:
            pn.regenerate_models()
        except Exception:
            pass
except Exception as _:
    pass

# If conductances are still NaN after model regeneration, compute a safe
# diffusive conductance array directly (D_avg * area / length) as a fallback.
try:
    g = np.asarray(air['throat.diffusive_conductance'])
    if np.count_nonzero(~np.isfinite(g)) > 0:
        print('Computing fallback throat.diffusive_conductance to replace NaNs...')
        conns = np.asarray(pn['throat.conns'])
        pdiff = np.asarray(air['pore.diffusivity']) if 'pore.diffusivity' in air.keys() else np.ones(pn.Np) * 1e-9
        td = np.asarray(pn['throat.diameter']) if 'throat.diameter' in pn.keys() else None
        tl = np.asarray(pn['throat.length'])
        # fallback sizes
        pos_td = td[np.isfinite(td) & (td > 0)] if td is not None else np.array([])
        if pos_td.size > 0:
            small_eps = float(np.min(pos_td) * 0.1)
        else:
            small_eps = 1e-9
        small_len = float(np.min(shape) * 1e-6)
        g_new = np.zeros(pn.Nt)
        for ii in range(pn.Nt):
            try:
                c = conns[ii]
                Davg = np.nanmean(pdiff[c]) if pdiff is not None else 1e-9
                Lval = float(tl[ii]) if np.isfinite(tl[ii]) and tl[ii] > 0 else small_len
                if td is not None and np.isfinite(td[ii]) and td[ii] > 0:
                    Aval = np.pi * (float(td[ii]) / 2.0) ** 2
                else:
                    # try to use pore diameters if present
                    try:
                        pdia = np.asarray(pn['pore.diameter'])
                        d0 = float(pdia[c[0]]) if (c[0] < pdia.size and pdia[c[0]]>0) else small_eps
                        d1 = float(pdia[c[1]]) if (c[1] < pdia.size and pdia[c[1]]>0) else small_eps
                        dmin = min(d0, d1)
                        Aval = np.pi * (dmin / 2.0) ** 2
                    except Exception:
                        Aval = np.pi * (small_eps / 2.0) ** 2
                g_new[ii] = Davg * Aval / Lval
            except Exception:
                g_new[ii] = 0.0
        air['throat.diffusive_conductance'] = g_new
except Exception:
    pass

# Quick diagnostics after regeneration
try:
    tl = np.asarray(pn['throat.length'])
    g = np.asarray(air['throat.diffusive_conductance'])
    print('Post-regenerate throat.length: min=', np.nanmin(tl), 'max=', np.nanmax(tl), 'n_nonpos=', np.count_nonzero(tl<=0))
    print('Post-regenerate throat.diffusive_conductance: finite=', np.count_nonzero(np.isfinite(g)), 'non-finite=', np.count_nonzero(~np.isfinite(g)))
except Exception as _:
    print('Could not print post-regeneration diagnostics (missing arrays)')

# Determine internal pores (those not in any boundary layer) in a robust way.
boundary_labels = ['left_boundary', 'right_boundary', 'front_boundary', 'back_boundary', 'top_boundary', 'bottom_boundary']
_b_list = []
for _lbl in boundary_labels:
    try:
        _arr = pn.pores(_lbl)
        if _arr.size > 0:
            _b_list.append(_arr)
    except Exception:
        # label may not exist on this network; skip
        pass
if len(_b_list) > 0:
    boundary_pores_all = np.unique(np.concatenate(_b_list))
else:
    boundary_pores_all = np.array([], dtype=int)
internal_pores = np.setdiff1d(pn.Ps, boundary_pores_all)

# Recompute face_internal sets to exclude any boundary-layer pores (they should be internal)
face_left_internal = np.intersect1d(face_left_internal, internal_pores)
face_right_internal = np.intersect1d(face_right_internal, internal_pores)

print('After excluding boundary-layer pores:')
print('Counts corrected:', 'left_internal', len(face_left_internal), 'right_internal', len(face_right_internal))
print('Sample corrected left_internal indices:', face_left_internal[:10])
print('Sample corrected right_internal indices:', face_right_internal[:10])

# --- 4) Setup algorithm and apply source/sink on the internal faces ---
fd = op.algorithms.FickianDiffusion(network=pn, phase=air)

# Create a source array on the phase (zeros everywhere)
air['pore.source_term'] = np.zeros(pn.Np)
# Distribute the total Q across the face pores so that the net inserted mass is balanced
try:
    if face_left_internal.size > 0 and face_right_internal.size > 0:
        # Interpret Q_total as the TOTAL flow to be inserted on the left face and removed on the right face
        val_left = float(Q_total) / float(face_left_internal.size)
        val_right = -float(Q_total) / float(face_right_internal.size)
        air['pore.source_term'][face_left_internal] = val_left
        air['pore.source_term'][face_right_internal] = val_right
    else:
        # fallback: per-pore assignment as before
        air['pore.source_term'][face_left_internal] = Q_total
        air['pore.source_term'][face_right_internal] = -Q_total
except Exception:
    # keep the previous conservative assignment if anything goes wrong
    air['pore.source_term'][face_left_internal] = Q_total
    air['pore.source_term'][face_right_internal] = -Q_total
# Attach source to algorithm (algorithm reads the array from the phase).
# Try to attach the source specifically to the internal pores we computed above.
try:
    fd.set_source(propname='pore.source_term', pores=internal_pores)
except TypeError:
    try:
        fd.set_source(propname='pore.source_term')
    except Exception:
        fd.set_source(propname='pore.source_term', pores=pn.Ps)

# Debug: inspect algorithm to ensure source attached
try:
    print('FickianDiffusion settings:', getattr(fd, 'settings', {}))
    src_attrs = [a for a in dir(fd) if 'source' in a.lower()]
    print('Attributes containing "source" on fd:', src_attrs)
    # print any public attribute that might store sources
    for name in ['sources', '_sources', 'source', 'source_list', 'sources_list']:
        if hasattr(fd, name):
            try:
                val = getattr(fd, name)
                print(f'fd.{name} ->', type(val), getattr(val, '__len__', lambda:None)())
            except Exception:
                print(f'fd.{name} present but could not print value')
except Exception:
    pass
# Anchor a single pore concentration to remove nullspace (provides reference)
try:
    if internal_pores.size > 0:
        fd.set_value(pores=[int(internal_pores[0])], values=0.0)
        print('Pinned pore', int(internal_pores[0]), 'to value 0.0 to anchor solution')
except Exception:
    pass

# Solve
print('Running FickianDiffusion solver...')

# --- Extra diagnostics (non-invasive) ---------------------------------
# Print the source-term on the face pores and basic checks to debug zero ΔC
try:
    s = np.asarray(air['pore.source_term'])
    print('Source-term sum (global):', float(np.nansum(s)))
    print('Source-term sample left sum/count:', float(np.nansum(s[face_left_internal])),
          face_left_internal.size)
    print('Source-term sample right sum/count:', float(np.nansum(s[face_right_internal])),
          face_right_internal.size)
except Exception as _:
    print('Could not inspect pore.source_term before solve')

# Inspect per-face-pore connectivity: sum of incident throat conductances
try:
    conns = np.asarray(pn['throat.conns'])
    g = np.asarray(air['throat.diffusive_conductance'])
    def pore_incident_conductance(p):
        # find throat indices where pore p is one of the connections
        idxs = np.where((conns[:, 0] == p) | (conns[:, 1] == p))[0]
        return float(np.nansum(g[idxs])) if idxs.size>0 else 0.0

    # show a few samples
    print('Sample incident conductance sums for left_internal (up to 10):',
          [pore_incident_conductance(int(p)) for p in face_left_internal[:10]])
    print('Sample incident conductance sums for right_internal (up to 10):',
          [pore_incident_conductance(int(p)) for p in face_right_internal[:10]])

    # Any throat directly connecting left_internal to right_internal?
    left_set = set(face_left_internal.tolist())
    right_set = set(face_right_internal.tolist())
    direct = []
    for ti in range(pn.Nt):
        a, b = int(conns[ti, 0]), int(conns[ti, 1])
        if (a in left_set and b in right_set) or (b in left_set and a in right_set):
            direct.append((ti, (a, b), float(g[ti]) if ti < g.size else None))
    print('Throats directly connecting left_internal <-> right_internal (count):', len(direct))
    if len(direct) > 0:
        print('Sample direct connections (up to 10):', direct[:10])
except Exception as _:
    print('Could not inspect throat connectivity/ conductances before solve')


def _sanitize_and_report(fd, pn, air):
    issues = []
    # Check throat diffusive conductance
    try:
        g = np.asarray(air['throat.diffusive_conductance'])
        n_bad = np.count_nonzero(~np.isfinite(g))
        if n_bad > 0:
            issues.append(f"throat.diffusive_conductance has {n_bad} non-finite entries; attempting fallback repair")
            bad_idx = np.where(~np.isfinite(g))[0]
            print('Sample non-finite throat.diffusive_conductance indices (up to 20):', bad_idx[:20])
            print('Sample values at those indices:', g[bad_idx[:20]])
            # Gather helpers
            try:
                t_conns = np.asarray(pn['throat.conns'])
            except Exception:
                t_conns = None
            try:
                t_len = np.asarray(pn['throat.length'])
            except Exception:
                t_len = None
            try:
                t_diam = np.asarray(pn['throat.diameter']) if 'throat.diameter' in pn.keys() else None
            except Exception:
                t_diam = None
            pd = None
            try:
                pd = np.asarray(air['pore.diffusivity'])
            except Exception:
                pd = None
            # Fallback: compute per-bad-throat diffusive conductance using pore diffusivities and geometry
            # Use conservative small fallbacks for diameters/lengths when needed
            small_len = float(np.min(shape) * 1e-6)
            pos_td = None
            try:
                if t_diam is not None:
                    pos_td = t_diam[np.isfinite(t_diam) & (t_diam > 0)]
            except Exception:
                pos_td = None
            if pos_td is not None and pos_td.size > 0:
                small_eps = float(np.min(pos_td) * 0.1)
            else:
                small_eps = 1e-9
            for ii in bad_idx:
                try:
                    con = t_conns[ii] if t_conns is not None else None
                    Davg = np.nanmean(pd[con]) if (pd is not None and con is not None) else 1e-9
                    Lval = float(t_len[ii]) if (t_len is not None and np.isfinite(t_len[ii]) and t_len[ii] > 0) else small_len
                    if t_diam is not None and np.isfinite(t_diam[ii]) and t_diam[ii] > 0:
                        Aval = np.pi * (float(t_diam[ii]) / 2.0) ** 2
                    else:
                        # try to use pore diameters if present
                        try:
                            pdia = np.asarray(pn['pore.diameter'])
                            d0 = float(pdia[con[0]]) if (con is not None and con[0] < pdia.size and pdia[con[0]]>0) else small_eps
                            d1 = float(pdia[con[1]]) if (con is not None and con[1] < pdia.size and pdia[con[1]]>0) else small_eps
                            dmin = min(d0, d1)
                            Aval = np.pi * (dmin / 2.0) ** 2
                        except Exception:
                            Aval = np.pi * (small_eps / 2.0) ** 2
                    g[ii] = Davg * Aval / Lval
                except Exception:
                    g[ii] = 0.0
            air['throat.diffusive_conductance'] = g
    except Exception:
        issues.append('Could not access throat.diffusive_conductance')

    # Check throat lengths (avoid zeros/negatives that lead to inf conductances)
    try:
        tl = np.asarray(pn['throat.length'])
        n_zero = np.count_nonzero(tl <= 0)
        if n_zero > 0:
            issues.append(f"throat.length has {n_zero} entries <= 0; repairing lengths and recomputing conductances for those throats")
            bad = np.where(tl <= 0)[0]
            print('Sample throat.length <=0 indices (up to 10):', bad[:10])
            print('Sample throat.length values at those indices:', tl[bad[:10]])
            # Replace non-positive lengths with a small physical length and write back to network
            small_len = float(np.min(shape) * 1e-6)
            tl[bad] = small_len
            pn['throat.length'] = tl
            # Recompute conductance for those throats using pore diffusivities and geometry
            try:
                g = np.asarray(air['throat.diffusive_conductance'])
            except Exception:
                g = np.zeros(pn.Nt)
            try:
                t_conns = np.asarray(pn['throat.conns'])
            except Exception:
                t_conns = None
            try:
                t_diam = np.asarray(pn['throat.diameter']) if 'throat.diameter' in pn.keys() else None
            except Exception:
                t_diam = None
            pd = None
            try:
                pd = np.asarray(air['pore.diffusivity'])
            except Exception:
                pd = None
            pos_td = None
            try:
                if t_diam is not None:
                    pos_td = t_diam[np.isfinite(t_diam) & (t_diam > 0)]
            except Exception:
                pos_td = None
            if pos_td is not None and pos_td.size > 0:
                small_eps = float(np.min(pos_td) * 0.1)
            else:
                small_eps = 1e-9
            for ii in bad:
                try:
                    con = t_conns[ii] if t_conns is not None else None
                    Davg = np.nanmean(pd[con]) if (pd is not None and con is not None) else 1e-9
                    Lval = float(tl[ii])
                    if t_diam is not None and np.isfinite(t_diam[ii]) and t_diam[ii] > 0:
                        Aval = np.pi * (float(t_diam[ii]) / 2.0) ** 2
                    else:
                        try:
                            pdia = np.asarray(pn['pore.diameter'])
                            d0 = float(pdia[con[0]]) if (con is not None and con[0] < pdia.size and pdia[con[0]]>0) else small_eps
                            d1 = float(pdia[con[1]]) if (con is not None and con[1] < pdia.size and pdia[con[1]]>0) else small_eps
                            dmin = min(d0, d1)
                            Aval = np.pi * (dmin / 2.0) ** 2
                        except Exception:
                            Aval = np.pi * (small_eps / 2.0) ** 2
                    g[ii] = Davg * Aval / Lval
                except Exception:
                    g[ii] = 0.0
            air['throat.diffusive_conductance'] = g
    except Exception:
        issues.append('Could not access throat.length')

    # Check pore source term
    try:
        s = np.asarray(air['pore.source_term'])
        n_bad = np.count_nonzero(~np.isfinite(s))
        if n_bad > 0:
            issues.append(f"pore.source_term has {n_bad} non-finite entries; setting them to 0")
            s[~np.isfinite(s)] = 0.0
            air['pore.source_term'] = s
    except Exception:
        issues.append('Could not access pore.source_term')

    # Check pore diffusivity
    try:
        pd = np.asarray(air['pore.diffusivity'])
        if pd.size == 0 or not np.all(np.isfinite(pd)):
            issues.append('pore.diffusivity contains non-finite values; replacing with 1e-9')
            pd = np.ones(pn.Np) * 1e-9
            air['pore.diffusivity'] = pd
    except Exception:
        issues.append('Could not access pore.diffusivity')

    # Try to build the linear system and inspect A and b if available
    try:
        fd._build_A_b()
        A = getattr(fd, 'A', None)
        b = getattr(fd, 'b', None)
        if A is not None:
            # sparse matrix data
            try:
                data = A.data
                nbad = np.count_nonzero(~np.isfinite(data))
                if nbad > 0:
                    issues.append(f'Linear system matrix A contains {nbad} non-finite entries')
            except Exception:
                pass
        if b is not None:
            try:
                nbad = np.count_nonzero(~np.isfinite(b))
                if nbad > 0:
                    issues.append(f'Linear system vector b contains {nbad} non-finite entries')
            except Exception:
                pass
    except Exception as e:
        issues.append(f'Could not build linear system for inspection: {e}')

    return issues


issues = _sanitize_and_report(fd, pn, air)
if len(issues) > 0:
    print('Pre-solve diagnostics found issues:')
    for it in issues:
        print('  -', it)
    print('Attempting to run solver after sanitization...')
try:
    fd.run()
except Exception as exc:
    print('Solver failed with error:', exc)
    print('Running diagnostics to help locate NaN/Inf sources...')
    # Re-run build and print any remaining problems
    try:
        fd._build_A_b()
        A = getattr(fd, 'A', None)
        b = getattr(fd, 'b', None)
        if A is not None:
            try:
                data = A.data
                bad_idx = np.where(~np.isfinite(data))[0]
                print('A.data non-finite count:', bad_idx.size)
                if bad_idx.size > 0:
                    print('Sample A.data[bad_idx[:10]] =', data[bad_idx[:10]])
            except Exception as e:
                print('Could not inspect A.data:', e)
        if b is not None:
            try:
                badb = np.where(~np.isfinite(b))[0]
                print('b non-finite count:', badb.size)
                if badb.size > 0:
                    print('Sample b[badb[:10]] =', b[badb[:10]])
            except Exception as e:
                print('Could not inspect b:', e)
    except Exception as e:
        print('Could not rebuild linear system for diagnostics:', e)
    raise
print('Solver finished.')

# --- 5) Postprocess: compute D_eff from imposed Q and ΔC across internal faces ---
# Use average concentrations on the internal faces
conc = fd['pore.concentration']
C_left = np.mean(conc[face_left_internal]) if face_left_internal.size > 0 else np.nan
C_right = np.mean(conc[face_right_internal]) if face_right_internal.size > 0 else np.nan

# Debug: show sample concentrations on face pores to diagnose zero ΔC
try:
    left_vals = conc[face_left_internal] if face_left_internal.size>0 else np.array([])
    right_vals = conc[face_right_internal] if face_right_internal.size>0 else np.array([])
    print('Sample concentrations on left_internal (up to 10):', left_vals[:10])
    print('Sample concentrations on right_internal (up to 10):', right_vals[:10])
    print('Non-finite counts left/right:', np.count_nonzero(~np.isfinite(left_vals)), np.count_nonzero(~np.isfinite(right_vals)))
    print('Non-zero counts left/right:', np.count_nonzero(np.abs(left_vals) > 0), np.count_nonzero(np.abs(right_vals) > 0))
except Exception as _:
    pass

delta_C = abs(C_left - C_right)
L = domain[axis]
A = domain[other_axes[0]] * domain[other_axes[1]]

print('C_left_avg =', C_left, 'C_right_avg =', C_right, 'delta_C =', delta_C)
if delta_C == 0 or A <= 0 or L <= 0:
    print('Cannot compute D_eff: zero delta_C or degenerate geometry')
    D_eff = float('nan')
else:
    # Q_total is the total source inserted on the face; use positive value
    D_eff = abs(Q_total) * L / (A * delta_C)
    print('D_eff =', D_eff)

# --- 6) Optional: save a quick 2D plot of concentration along flow axis ---
try:
    z = coords[internal_pores, axis]
    y = conc[internal_pores]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(z, y, s=8, alpha=0.6)
    ax.set_xlabel('Position along flow axis (m)')
    ax.set_ylabel('Pore concentration')
    ax.set_title('Concentration along flow axis (internal pores)')
    fig.savefig('periodic_conc_vs_pos.png', dpi=200, bbox_inches='tight')
    print('Saved periodic_conc_vs_pos.png')
except Exception as e:
    print('Could not save plot:', e)

print('Done.')
print('Done.')

# ------------------------------------------------------------------
# Extended: compute D_eff and tortuosity along all principal axes (x,y,z)
# This block reuses the same network and phase. It creates axis-specific
# periodic pairings only if they don't already exist and runs the same
# balanced-source procedure to compute D_eff and tortuosity per axis.
# ------------------------------------------------------------------

def _compute_Deff_and_tortuosity_for_axis(ax, Q=Q_total, create_periodic=True):
    left_internal, right_internal = _face_internal_pores_for_axis(ax)

    # labels for boundary layers per axis
    labels = {0: ('left_boundary', 'right_boundary'),
              1: ('front_boundary', 'back_boundary'),
              2: ('top_boundary', 'bottom_boundary')}
    b_left_label, b_right_label = labels[ax]
    b_left = pn.pores(b_left_label)
    b_right = pn.pores(b_right_label)

    if create_periodic:
        pair_and_connect(right_internal, b_left, axes=tuple(i for i in (0, 1, 2) if i != ax), label=f'periodic_axis{ax}')
        pair_and_connect(left_internal, b_right, axes=tuple(i for i in (0, 1, 2) if i != ax), label=f'periodic_axis{ax}')

    pn.regenerate_models()
    air.regenerate_models()

    fd_local = op.algorithms.FickianDiffusion(network=pn, phase=air)

    # balanced source on internal faces (distribute total Q across face pores)
    air['pore.source_term'] = np.zeros(pn.Np)
    try:
        if left_internal.size > 0 and right_internal.size > 0:
            val_left = float(Q) / float(left_internal.size)
            val_right = -float(Q) / float(right_internal.size)
            air['pore.source_term'][left_internal] = val_left
            air['pore.source_term'][right_internal] = val_right
        else:
            air['pore.source_term'][left_internal] = Q
            air['pore.source_term'][right_internal] = -Q
    except Exception:
        air['pore.source_term'][left_internal] = Q
        air['pore.source_term'][right_internal] = -Q
    try:
        fd_local.set_source(propname='pore.source_term', pores=np.concatenate([left_internal, right_internal]) if left_internal.size+right_internal.size>0 else pn.Ps)
    except TypeError:
        try:
            fd_local.set_source(propname='pore.source_term')
        except Exception:
            fd_local.set_source(propname='pore.source_term', pores=pn.Ps)

    # Anchor one pore to remove singular nullspace (reference)
    try:
        if left_internal.size + right_internal.size > 0:
            ref = int(left_internal[0]) if left_internal.size>0 else int(right_internal[0])
            fd_local.set_value(pores=[ref], values=0.0)
            print(f'Pinned pore {ref} to 0.0 for axis {ax} run')
    except Exception:
        pass

    issues = _sanitize_and_report(fd_local, pn, air)
    if issues:
        print(f'Pre-solve diagnostics for axis {ax}:')
        for it in issues:
            print('  -', it)
    fd_local.run()

    conc = fd_local['pore.concentration']
    C_left = np.mean(conc[left_internal]) if left_internal.size > 0 else np.nan
    C_right = np.mean(conc[right_internal]) if right_internal.size > 0 else np.nan
    delta_C = abs(C_left - C_right)
    L = domain[ax]
    other = [i for i in (0, 1, 2) if i != ax]
    A = domain[other[0]] * domain[other[1]]

    if delta_C == 0 or A <= 0 or L <= 0 or np.isnan(delta_C):
        D_eff_ax = float('nan')
    else:
        D_eff_ax = abs(Q) * L / (A * delta_C)

    # porosity
    V_p = np.sum(pn['pore.volume']) if 'pore.volume' in pn.keys() else 0.0
    V_t = np.sum(pn['throat.volume']) if 'throat.volume' in pn.keys() else 0.0
    V_bulk = float(np.prod(domain))
    porosity = (V_p + V_t) / V_bulk if V_bulk > 0 else float('nan')

    try:
        D_AB = np.mean(np.asarray(air['pore.diffusivity']))
    except Exception:
        D_AB = float('nan')

    tau = porosity * D_AB / D_eff_ax if D_eff_ax and np.isfinite(D_eff_ax) else float('nan')
    return D_eff_ax, tau


results = {}
for ax in (0, 1, 2):
    print(f'Computing D_eff and tortuosity for axis {ax}...')
    try:
        Deff_ax, tau_ax = _compute_Deff_and_tortuosity_for_axis(ax)
    except Exception as e:
        print(f'Axis {ax} computation failed: {e}')
        Deff_ax, tau_ax = float('nan'), float('nan')
    results[ax] = (Deff_ax, tau_ax)
    print(f'Axis {ax}: D_eff = {Deff_ax}, tortuosity = {tau_ax}')

print('Summary (axis: D_eff, tortuosity):')
for ax, (d, t) in results.items():
    print(f'  axis {ax}: D_eff={d}, tau={t}')
