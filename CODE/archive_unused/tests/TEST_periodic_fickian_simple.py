##THIS THE CURRENT TESTING FILE
#I CANT FIND THE RIGHT FUCKING ONE

import numpy as np
import openpnm as op
from scipy.spatial import KDTree
from scipy.sparse.linalg import lsqr
from scipy.sparse import csr_matrix
from NETWORK_GENERATION_FINAL import build_voronoi_network

# ---- User parameters ----
domain = (250e-8, 250e-8, 250e-8)
seed = 42
pore_density = 1
axis = 0               # flow axis: 0=x,1=y,2=z
Q_total = 1.0          # total source inserted on left face, removed on right face

# ---- 1) Build network ----
pn = build_voronoi_network(domain_size=domain, pore_density=pore_density,
                           points=None, seed=seed, save_files=False)
coords = np.asarray(pn['pore.coords'])
shape = np.asarray(domain)

# ---- 2) Identify face internal pores and boundary layer pores ----

tol = 1e-12 ##change this to be dependent on doamin size/density etc... (ie investigate)
left_internal = pn.Ps[coords[:, axis] < (0 + tol)]
right_internal = pn.Ps[coords[:, axis] > (shape[axis] - tol)]

left_boundary = pn.pores('left_boundary')
right_boundary = pn.pores('right_boundary')

boundary_labels = ['left_boundary', 'right_boundary', 'front_boundary',
                   'back_boundary', 'top_boundary', 'bottom_boundary']
b_list = []
for lbl in boundary_labels:
    try:
        arr = pn.pores(lbl)
        if arr.size:
            b_list.append(arr)
    except Exception:
        pass
boundary_all = np.unique(np.concatenate(b_list)) if b_list else np.array([], dtype=int)
internal_pores = np.setdiff1d(pn.Ps, boundary_all)

left_internal = np.intersect1d(left_internal, internal_pores)
right_internal = np.intersect1d(right_internal, internal_pores)

# ---- 3) Pairing function using transverse KDTree and connect periodic throats ---- ##tHIS IS THE MAIN ISSUE!!!
other_axes = tuple(i for i in (0, 1, 2) if i != axis)

def pair_and_connect_kdtree(internal_idxs, boundary_idxs, axes=other_axes, label='periodic'):
    if internal_idxs.size == 0 or boundary_idxs.size == 0:
        return np.array([], dtype=int)
    b_coords = coords[boundary_idxs][:, axes]
    tree = KDTree(b_coords)
    i_coords = coords[internal_idxs][:, axes]
    _, idxs = tree.query(i_coords)
    paired = boundary_idxs[idxs]
    full_vec = coords[paired] - coords[internal_idxs]
    min_sep = float(np.min(shape)) * 1e-12
    valid_mask = np.linalg.norm(full_vec, axis=1) > min_sep
    internal_sel = internal_idxs[valid_mask]
    boundary_sel = paired[valid_mask]
    existing_conns = np.asarray(pn['throat.conns'])
    exists_check = lambda a,b: np.any(np.all(existing_conns == [a, b], axis=1) | np.all(existing_conns == [b, a], axis=1)) if existing_conns.size>0 else False
    for p1, p2 in zip(internal_sel, boundary_sel):
        if exists_check(int(p1), int(p2)):
            continue
        op.topotools.connect_pores(network=pn, pores1=[int(p1)], pores2=[int(p2)], labels=[label])
    return pn.throats(label)

pair_and_connect_kdtree(right_internal, left_boundary, axes=other_axes, label='periodic')
pair_and_connect_kdtree(left_internal, right_boundary, axes=other_axes, label='periodic')

# ---- 4) Compute correct wrapped throat lengths for periodic throats ----
periodic_tids = pn.throats('periodic')
if periodic_tids.size:
    conns = np.asarray(pn['throat.conns'])[periodic_tids]
    p0 = coords[conns[:, 0]]
    p1 = coords[conns[:, 1]]
    vecs = p1 - p0
    for dim in (0, 1, 2):
        L = shape[dim]
        if L > 0:
            mask = np.abs(vecs[:, dim]) > (L / 2.0)
            vecs[mask, dim] -= np.sign(vecs[mask, dim]) * L
    lengths = np.linalg.norm(vecs, axis=1)
    pn['throat.length'][periodic_tids] = lengths

# ---- 5) Phase, physics, regenerate models ----
air = op.phase.Air(network=pn)
air.add_model_collection(op.models.collections.physics.basic)
pn.regenerate_models()
air.regenerate_models()

# ---- 5b) Ensure diffusive conductance exists (fallback) ----
if 'throat.diffusive_conductance' not in air.keys():
    D = np.ones(pn.Np) * 1e-9
    try:
        D = np.asarray(air['pore.diffusivity'])
    except Exception:
        pass
    td = np.asarray(pn['throat.diameter']) if 'throat.diameter' in pn.keys() else None
    conns = np.asarray(pn['throat.conns'])
    Nt = pn.Nt
    g = np.zeros(Nt)
    small_d = 1e-9
    small_L = float(np.min(shape) * 1e-6)
    tlengths = np.asarray(pn['throat.length'])
    for ti in range(Nt):
        Lval = float(tlengths[ti]) if np.isfinite(tlengths[ti]) and tlengths[ti] > 0 else small_L
        if td is not None and np.isfinite(td[ti]) and td[ti] > 0:
            A = np.pi * (float(td[ti]) / 2.0) ** 2
        else:
            try:
                pd = np.asarray(pn['pore.diameter'])
                c = conns[ti]
                d0 = pd[int(c[0])] if int(c[0]) < pd.size and pd[int(c[0])] > 0 else small_d
                d1 = pd[int(c[1])] if int(c[1]) < pd.size and pd[int(c[1])] > 0 else small_d
                A = np.pi * (min(d0, d1) / 2.0) ** 2
            except Exception:
                A = np.pi * (small_d / 2.0) ** 2
        Davg = np.nanmean(D[conns[ti]]) if D.size == pn.Np else np.nanmean(D)
        g[ti] = Davg * A / Lval
    air['throat.diffusive_conductance'] = g

# ---- 6) Balanced-source Fickian diffusion ----
fd = op.algorithms.FickianDiffusion(network=pn, phase=air)

air['pore.source_term'] = np.zeros(pn.Np)
if left_internal.size and right_internal.size:
    air['pore.source_term'][left_internal] = float(Q_total) / float(left_internal.size)
    air['pore.source_term'][right_internal] = -float(Q_total) / float(right_internal.size)
else:
    air['pore.source_term'][internal_pores] = float(Q_total) / float(internal_pores.size) if internal_pores.size else 0.0

# pick pinned pore not in source faces, zero its source, attach sources excluding it
candidates = np.setdiff1d(internal_pores, np.concatenate([left_internal, right_internal]))
pinned = int(candidates[0]) if candidates.size else (int(internal_pores[0]) if internal_pores.size else None)
if pinned is not None:
    air['pore.source_term'][pinned] = 0.0
    src_pores = np.setdiff1d(internal_pores, np.array([pinned], dtype=int))
else:
    src_pores = internal_pores

fd.set_source(propname='pore.source_term', pores=src_pores)
if pinned is not None:
    fd.set_value_BC(pores=[pinned], values=0.0)

# ----- Method A: locate rows with NaN/Inf, inspect incident throats, and repair only those throats -----
def build_linear_system(alg):
    # use available builder
    if hasattr(alg, "_build_A"):
        alg._build_A()
    elif hasattr(alg, "_build_A_b"):
        alg._build_A_b()
    else:
        raise RuntimeError("No builder method on algorithm")
    return alg.A, alg.b

A, b = build_linear_system(fd)
bad_idx = np.where(~np.isfinite(A.data))[0]
bad_b = np.where(~np.isfinite(b))[0]
print('Initial non-finite counts: A.data=', bad_idx.size, ', b=', bad_b.size)

if bad_idx.size == 0 and bad_b.size == 0:
    print('A and b finite. Running solver.')
    fd.run()
else:
    # Map bad data indices -> rows
    A_csr = A.tocsr()
    bad_rows = set()
    # map by checking each row segment for any bad data index
    bad_set = set(bad_idx.tolist())
    for r in range(A_csr.shape[0]):
        s, e = A_csr.indptr[r], A_csr.indptr[r+1]
        if s < e:
            row_range = set(range(s, e))
            if row_range & bad_set:
                bad_rows.add(r)
    bad_rows = np.array(sorted(list(bad_rows)), dtype=int)
    print('Bad rows detected:', bad_rows.size, ' sample:', bad_rows[:50])

    # Inspect and flag incident throats for repair
    conns = np.asarray(pn['throat.conns'])
    Nt = pn.Nt
    tl = np.asarray(pn['throat.length']) if 'throat.length' in pn.keys() else None
    td = np.asarray(pn['throat.diameter']) if 'throat.diameter' in pn.keys() else None
    g  = np.asarray(air['throat.diffusive_conductance']) if 'throat.diffusive_conductance' in air.keys() else None
    pd = np.asarray(air['pore.diffusivity']) if 'pore.diffusivity' in air.keys() else None
    psrc = np.asarray(air['pore.source_term']) if 'pore.source_term' in air.keys() else None
    coords = np.asarray(pn['pore.coords'])

    pore_to_throats = [[] for _ in range(pn.Np)]
    for ti in range(Nt):
        a, b_ = int(conns[ti,0]), int(conns[ti,1])
        pore_to_throats[a].append(ti)
        pore_to_throats[b_].append(ti)

    repair_throats = set()
    for p in bad_rows:
        inc = pore_to_throats[int(p)]
        # print summary for user
        print(f'\nBad pore row {p}, coord={coords[int(p)].tolist()}, incident_throats={len(inc)}')
        if psrc is not None:
            print('  pore.source_term=', float(psrc[int(p)]))
        if pd is not None:
            print('  pore.diffusivity=', float(pd[int(p)]))
        for ti in inc:
            Lval = float(tl[ti]) if (tl is not None and np.isfinite(tl[ti])) else None
            Dval = float(td[ti]) if (td is not None and np.isfinite(td[ti])) else None
            Gval = float(g[ti]) if (g is not None and np.isfinite(g[ti])) else None
            print(f'   throat {ti}: conns={conns[ti].tolist()} length={Lval} diam={Dval} g={Gval}')
            if (Lval is None) or (not np.isfinite(Lval)) or (Lval <= 0) or (Gval is None) or (not np.isfinite(Gval)):
                repair_throats.add(int(ti))

    repair_throats = np.array(sorted(list(repair_throats)), dtype=int)
    print('\nFlagged throats for repair:', repair_throats.size, ' sample:', repair_throats[:200])

    # Conservative repair of flagged throats
    if repair_throats.size > 0:
        small_len = float(np.min(shape) * 1e-6)
        small_d = 1e-9
        # ensure arrays exist
        if tl is None:
            tl = np.zeros(Nt) + small_len
        if td is None:
            td = np.zeros(Nt) + small_d
        # repair values only where needed
        for ti in repair_throats:
            if not (np.isfinite(tl[ti]) and tl[ti] > 0):
                tl[ti] = small_len
            if not (np.isfinite(td[ti]) and td[ti] > 0):
                td[ti] = small_d
        pn['throat.length'] = tl
        pn['throat.diameter'] = td

        # recompute conductances only for flagged throats
        Dp = np.ones(pn.Np) * 1e-9
        try:
            Dp = np.asarray(air['pore.diffusivity'])
        except Exception:
            pass
        g_arr = np.asarray(air['throat.diffusive_conductance']) if 'throat.diffusive_conductance' in air.keys() else np.zeros(Nt)
        for ti in repair_throats:
            try:
                c = conns[ti]
                Lval = float(tl[ti]) if np.isfinite(tl[ti]) and tl[ti] > 0 else small_len
                A_area = np.pi * (float(td[ti]) / 2.0) ** 2
                Davg = np.nanmean(Dp[c]) if Dp.size == pn.Np else np.nanmean(Dp)
                g_arr[ti] = Davg * A_area / Lval
            except Exception:
                g_arr[ti] = 1e-20
        air['throat.diffusive_conductance'] = g_arr

    # Regenerate and rebuild
    print('Regenerating models and rebuilding linear system after repairs...')
    pn.regenerate_models()
    air.regenerate_models()
    A2, b2 = build_linear_system(fd)
    bad_idx2 = np.count_nonzero(~np.isfinite(A2.data))
    bad_b2 = np.count_nonzero(~np.isfinite(b2))
    print('After repairs non-finite counts: A.data=', bad_idx2, ', b=', bad_b2)

    # Final check
    if bad_idx2 == 0 and bad_b2 == 0:
        print('Repairs removed non-finite entries. Running solver.')
        fd.run()
    else:
        # Map remaining bad indices to rows for user debugging and abort. Do NOT sanitize silently.
        print('Repairs did not remove all non-finite entries. Detailed mapping follows.')
        A2_csr = A2.tocsr()
        bad_data_idx2 = np.where(~np.isfinite(A2.data))[0]
        bad_rows2 = set()
        bad_set2 = set(bad_data_idx2.tolist())
        for r in range(A2_csr.shape[0]):
            s, e = A2_csr.indptr[r], A2_csr.indptr[r+1]
            if s < e and set(range(s, e)) & bad_set2:
                bad_rows2.add(r)
        bad_rows2 = np.array(sorted(list(bad_rows2)), dtype=int)
        print('Remaining bad rows count:', bad_rows2.size, ' sample up to 200:', bad_rows2[:200])
        # Print the first 100 bad A.data indices and their mapped rows
        print('Sample bad A.data indices (up to 200):', bad_data_idx2[:200])
        # Provide incident throat listing so you can inspect coordinates offline
        conns = np.asarray(pn['throat.conns'])
        coords = np.asarray(pn['pore.coords'])
        for p in bad_rows2[:200]:
            inc = [int(t) for t in (np.where((conns[:,0]==p)|(conns[:,1]==p))[0])]
            print(f'Row {p}: coord={coords[p].tolist()} incident_throats={len(inc)} sample_throats={inc[:20]}')
        raise Exception('Unresolved NaN/Inf in linear system after method A repairs. See printed mapping above.')

# ---- 7) Postprocess: D_eff from Q and deltaC ----
conc = fd['pore.concentration']
C_left = np.mean(conc[left_internal]) if left_internal.size else np.nan
C_right = np.mean(conc[right_internal]) if right_internal.size else np.nan
delta_C = abs(C_left - C_right)
L = domain[axis]
other = [i for i in (0, 1, 2) if i != axis]
Area = domain[other[0]] * domain[other[1]]

if delta_C == 0 or Area <= 0 or L <= 0 or np.isnan(delta_C):
    D_eff = float('nan')
else:
    D_eff = abs(Q_total) * L / (Area * delta_C)

print('D_eff =', D_eff)
