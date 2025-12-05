import numpy as np
import porespy as ps
import openpnm as op

op.visualization.set_mpl_style()

# small test to show D_eff scales linearly with molecular diffusivity D_AB
# but D_eff_norm = D_eff / D_AB is invariant

shape = (30, 30, 15)
voxel = 1e-5
porosity = 0.6
blobiness = 2
seed = 12345

np.random.seed(seed)
im = ps.generators.blobs(shape=shape, porosity=porosity, blobiness=blobiness)

snow = ps.networks.snow2(im, voxel_size=voxel)
net = op.io.network_from_porespy(snow.network)

# geometry
geo = op.models.collections.geometry.spheres_and_cylinders
net.add_model_collection(geo, domain='all')
net.regenerate_models()

# boundary helper
def _boundary_pores_by_coord(network, axis=0):
    coords = network['pore.coords']
    mn = coords[:, axis].min()
    mx = coords[:, axis].max()
    inlet = np.where(np.isclose(coords[:, axis], mn))[0]
    outlet = np.where(np.isclose(coords[:, axis], mx))[0]
    return inlet, outlet

# set BCs values
C_in, C_out = 10, 5

D_list = [1e-5, 2e-5]
results = []
for D in D_list:
    phase = op.phase.Phase(network=net)
    phase['pore.diffusivity'] = np.ones(net.Np) * D
    # attach basic physics
    phys = dict(op.models.collections.physics.basic)
    phys.pop('throat.entry_pressure', None)
    phase.add_model_collection(phys)
    phase.regenerate_models()

    fd = op.algorithms.FickianDiffusion(network=net, phase=phase)
    try:
        inlet = net.pores('left')
        outlet = net.pores('right')
    except Exception:
        inlet, outlet = _boundary_pores_by_coord(net, axis=0)

    fd.set_value_BC(pores=inlet, values=C_in)
    fd.set_value_BC(pores=outlet, values=C_out)
    fd.run()
    rate_inlet = float(fd.rate(pores=inlet)[0])
    # compute D_eff similarly to main script
    A = shape[1] * shape[2] * voxel ** 2
    L = shape[0] * voxel
    D_eff = rate_inlet * L / (A * (C_in - C_out))
    D_eff_norm = D_eff / D
    results.append((D, D_eff, D_eff_norm))

for D, Deff, Deff_norm in results:
    print(f"D_AB = {D:.1e} m^2/s -> D_eff = {Deff:.6e} m^2/s, D_eff_norm = {Deff_norm:.6e} (dimensionless)")

# report ratio of D_eff for the two runs (should be ~2)
ratio = results[1][1] / results[0][1] if results[0][1] != 0 else float('nan')
print(f"D_eff ratio (D=2e-5 / D=1e-5) = {ratio:.6f}")

# Cleanup
try:
    op.workspace.clear()
except Exception:
    pass

print('Test finished')
