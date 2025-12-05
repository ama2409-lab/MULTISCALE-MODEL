#this may work - explore application with voronoi
import openpnm as op
import numpy as np

# 1) Create cubic network
shape = [3, 3, 3]
spacing = 1e-5
axis =0
net = op.network.Cubic(shape=shape, spacing=spacing)
coords = net['pore.coords']

# 2) Identify left/right faces
left_face = net.pores('left')
right_face = net.pores('right')
left_sorted = left_face[np.lexsort((coords[left_face,2], coords[left_face,1]))]
right_sorted = right_face[np.lexsort((coords[right_face,2], coords[right_face,1]))]

# 3) Connect periodic pores
L = shape[axis] * spacing
for pL, pR in zip(left_sorted, right_sorted):
    tid = op.topotools.connect_pores(net, pores1=[pL], pores2=[pR], labels=['periodic'])

# 4) Add geometry: spheres + cylinders
net.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
net.regenerate_models()
# This will calculate throat.length and throat.diameter automatically for all throats

# 5) Add phase and physics
air = op.phase.Air(network=net)
air.add_model_collection(op.models.collections.physics.basic)
air.regenerate_models()

# ---------- balanced source approach (replace pin/fd.run) ----------
# Partition domain in x and apply +Q / -Q (periodic compatible)
coords = net['pore.coords']
axis = 0
x = coords[:, axis]
mid = (x.min() + x.max())/2.0
src_mask = x < mid
sink_mask = ~src_mask
Q_total = 1.0  # arbitrary scaling; smaller -> better conditioning sometimes

air['pore.source_term'] = np.zeros(net.Np, dtype=float)
air['pore.source_term'][src_mask]  =  Q_total / float(src_mask.sum())
air['pore.source_term'][sink_mask] = -Q_total / float(sink_mask.sum())

# Clean conductances (avoid NaN/Inf/zero)
g = np.asarray(air['throat.diffusive_conductance'])
g[~np.isfinite(g)] = 0.0
g[g <= 0] = 0.0
air['throat.diffusive_conductance'] = g

# algorithm setup
fd = op.algorithms.FickianDiffusion(network=net, phase=air)
nonzero_pores = np.where(air['pore.source_term'] != 0.0)[0]
fd.set_source(propname='pore.source_term', pores=nonzero_pores)

# numeric anchor only: pin a single isolated pore if needed (not a physical BC)
# but do NOT pin a face or many pores
isolated = np.where((np.abs((net['throat.conns'][:,0,None] == np.arange(net.Np)).sum(axis=0) +
                            (net['throat.conns'][:,1,None] == np.arange(net.Np)).sum(axis=0)) == 0))[0]
if isolated.size:
    fd.set_value_BC(pores=isolated.tolist(), values=0.0, mode='overwrite')
else:
    fd.set_value_BC(pores=[0], values=0.0)  # numerical anchor only

fd.run()

# ---------- compute D_eff ----------
C = fd['pore.concentration']
C_src = C[src_mask].mean()
C_sink = C[sink_mask].mean()
deltaC = C_src - C_sink
L = (net['pore.coords'][:,axis].max() - net['pore.coords'][:,axis].min()) + 0.0
cross_area = ( (net['pore.coords'][:, [i for i in (0,1,2) if i!=axis]].ptp(axis=0)).prod() )
D_eff = abs(Q_total) * L / (cross_area * abs(deltaC)) if deltaC != 0 else np.nan

print("C_src, C_sink, deltaC:", C_src, C_sink, deltaC)
print("D_eff:", D_eff)

# ---------- Improved 3D plotting with colormap and periodic ghost image ----------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')

# plot throats (light grey)
op.visualization.plot_connections(net, ax=ax, color='lightgrey', linewidth=0.7)

# plot pores colored by concentration
pc = C
op.visualization.plot_coordinates(net, ax=ax, color_by=pc, size_by=net['pore.diameter'], markersize=120)

# create a ghost image shifted by +L to visualize periodic mapping along x
Lbox = (net['pore.coords'][:,axis].max() - net['pore.coords'][:,axis].min())
coords_ghost = net['pore.coords'].copy()
coords_ghost[:, axis] += Lbox
# draw ghost pores faintly
op.visualization.plot_coordinates(coords=coords_ghost, ax=ax, color='lightgrey', markersize=60)

# colorbar: create scatter proxy to show full colormap range
sc = ax.scatter(net['pore.coords'][:,0], net['pore.coords'][:,1], net['pore.coords'][:,2],
                c=pc, s=0)   # invisible points to get mappable
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, shrink=0.6)
cbar.set_label('Concentration')

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('3D network concentration (with periodic ghost)')
plt.tight_layout()
plt.show()

