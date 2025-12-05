import openpnm as op
import numpy as np
from NETWORK_GENERATION_FINAL import build_voronoi_network

print('Diagnostic: building Voronoi network')

domain = (250e-6,250e-6,250e-6)
density = 1e13
num_points = int(max(10, round(density * np.prod(domain))))
pn = build_voronoi_network(domain_size=domain, pore_density=density, points=num_points, seed=42, save_files=False)
print(f'Network: Np={pn.Np}, Nt={pn.Nt}')

air = op.phase.Air(network=pn)
phys = op.models.collections.physics.basic
try:
    del phys['throat.entry_pressure']
except Exception:
    pass
air.add_model_collection(phys)
air.regenerate_models()

# conservative repairs as in the main script
import numpy as _np
print('Running conservative repairs...')
# fix throat diameters
td = _np.asarray(pn['throat.diameter'], dtype=float)
bad_td = (~_np.isfinite(td)) | (td <= 0)
print('bad_td count =', _np.sum(bad_td))
if _np.any(bad_td):
    pos = td[td > 0]
    min_pos_td = float(_np.min(pos)) if pos.size>0 else 1e-9
    replacement_td = max(min_pos_td * 1e-3, 1e-9)
    td[bad_td] = replacement_td
    pn['throat.diameter'] = td
    print('replaced bad_td with', replacement_td)

# fix throat lengths
tl = _np.asarray(pn['throat.length'], dtype=float)
bad_tl = (~_np.isfinite(tl)) | (tl <= 0)
print('bad_tl count =', _np.sum(bad_tl))
if _np.any(bad_tl):
    pos = tl[tl>0]
    if pos.size>0:
        min_pos_tl = float(_np.min(pos))
    else:
        coords = _np.asarray(pn['pore.coords'])
        conns = _np.asarray(pn['throat.conns'])
        dists = _np.linalg.norm(coords[conns[:,0]] - coords[conns[:,1]], axis=1)
        min_pos_tl = float(_np.min(dists[dists>0])) if _np.any(dists>0) else 1e-9
    replacement_tl = max(min_pos_tl * 1e-3, 1e-9)
    tl[bad_tl] = replacement_tl
    pn['throat.length'] = tl
    print('replaced bad_tl with', replacement_tl)

pn.regenerate_models()
air.regenerate_models()

# compute conductances
g = _np.asarray(air.get('throat.diffusive_conductance', _np.zeros(pn.Nt)), dtype=float)
print('conductance stats: finite=', _np.sum(_np.isfinite(g)), 'positive=', _np.sum(g>0), 'zeros=', _np.sum(g==0))
print('g min/max', _np.nanmin(g), _np.nanmax(g))

# compute isolated pores
incident = _np.zeros(pn.Np, dtype=float)
for i,(a,b) in enumerate(pn['throat.conns']):
    incident[a] += float(g[i])
    incident[b] += float(g[i])
isolated = _np.where(incident==0)[0]
print('isolated pores count =', isolated.size)

# porosity check
pv = _np.asarray(pn.get('pore.volume', _np.array([])), dtype=float)
if pv.size==0 or not _np.all(_np.isfinite(pv)):
    pd = _np.asarray(pn['pore.diameter'], dtype=float)
    pv = (np.pi/6.0) * pd**3
    print('computed pore.volume from diameters')

tv = _np.asarray(pn.get('throat.volume', _np.array([])), dtype=float)
if tv.size==0 or not _np.all(_np.isfinite(tv)):
    td = _np.asarray(pn['throat.diameter'], dtype=float)
    tl = _np.asarray(pn['throat.length'], dtype=float)
    tv = np.pi * (td/2.0)**2 * tl
    print('computed throat.volume from diameter/length')

V_p = pv.sum()
V_t = tv.sum()
V_bulk = _np.prod(domain)
porosity = (V_p + V_t) / V_bulk
print('porosity =', porosity)

print('Done diagnostics')
