import numpy as np
import porespy as ps
import openpnm as op
import time
import pandas as pd

op.visualization.set_mpl_style()

# This script creates one microstructure at a fixed physical size and
# tests two voxelizations (fine and coarse) that represent the same
# physical volume. It runs the diffusion pipeline and records D_eff
# and D_eff_norm and correlation length (voxels and meters).

shape_fine = (60, 60, 30)   # fine grid
shape_coarse = (30, 30, 15) # coarse grid (exactly half resolution -> same physical size)
physical_size = np.array([60, 60, 30]) * 1e-5  # meters; corresponds to fine voxel=1e-5

porosity = 0.6
blobiness = 2
seed = 123

np.random.seed(seed)
# generate a fine-grained image
im_fine = ps.generators.blobs(shape=shape_fine, porosity=porosity, blobiness=blobiness)

# downsample fine image to coarse by simple block reduction (average and threshold)
# to keep the same physical size when voxel doubles
shape_fine_arr = np.array(shape_fine)
shape_coarse_arr = np.array(shape_coarse)
assert np.all(shape_fine_arr % shape_coarse_arr == 0), "Shapes not integer multiples"
factor = shape_fine_arr // shape_coarse_arr

# block reduce by mean and threshold at 0.5
im_coarse = im_fine.reshape(shape_coarse[0], factor[0], shape_coarse[1], factor[1], shape_coarse[2], factor[2]).mean(axis=(1,3,5)) > 0.5

# define a helper to run pipeline and return metrics

def run_pipeline(im, voxel, shape):
    snow = ps.networks.snow2(im, voxel_size=voxel)
    net = op.io.network_from_porespy(snow.network)
    geo = op.models.collections.geometry.spheres_and_cylinders
    net.add_model_collection(geo, domain='all')
    net.regenerate_models()
    phase = op.phase.Phase(network=net)
    phase['pore.diffusivity'] = np.ones(net.Np) * 1e-5
    phys = dict(op.models.collections.physics.basic)
    phys.pop('throat.entry_pressure', None)
    phase.add_model_collection(phys)
    phase.regenerate_models()
    fd = op.algorithms.FickianDiffusion(network=net, phase=phase)
    try:
        inlet = net.pores('left')
        outlet = net.pores('right')
    except Exception:
        coords = net['pore.coords']
        mn = coords[:, 0].min(); mx = coords[:, 0].max()
        inlet = np.where(np.isclose(coords[:, 0], mn))[0]
        outlet = np.where(np.isclose(coords[:, 0], mx))[0]
    C_in, C_out = 10, 5
    fd.set_value_BC(pores=inlet, values=C_in)
    fd.set_value_BC(pores=outlet, values=C_out)
    fd.run()
    rate_inlet = float(fd.rate(pores=inlet)[0])
    A = shape[1] * shape[2] * voxel ** 2
    L = shape[0] * voxel
    D_eff = rate_inlet * L / (A * (C_in - C_out))
    D_AB = float(phase['pore.diffusivity'][0])
    D_eff_norm = D_eff / D_AB if D_AB != 0 else float('nan')
    # correlation length (radial) in voxels
    arr = im.astype(float)
    arr -= arr.mean()
    f = np.fft.fftn(arr)
    power = np.abs(f) ** 2
    corr = np.fft.ifftn(power).real
    corr = np.fft.fftshift(corr)
    nx, ny, nz = arr.shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    x = np.arange(nx) - cx
    y = np.arange(ny) - cy
    z = np.arange(nz) - cz
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2).ravel()
    corr_r = corr.ravel()
    maxr = int(np.ceil(R.max()))
    radial = np.zeros(maxr + 1)
    counts = np.zeros_like(radial)
    inds = np.floor(R).astype(int)
    for idx_val, val in enumerate(inds):
        radial[val] += corr_r[idx_val]
        counts[val] += 1
    radial = radial / np.maximum(counts, 1)
    radial_norm = radial / (radial[0] if radial[0] != 0 else 1.0)
    idxs = np.where(radial_norm <= np.exp(-1))[0]
    corr_length_vox = float(idxs[0]) if idxs.size > 0 else float(maxr)
    corr_length_m = corr_length_vox * voxel
    return {'D_eff': D_eff, 'D_eff_norm': D_eff_norm, 'corr_length_vox': corr_length_vox, 'corr_length_m': corr_length_m}

# run fine
voxel_fine = 1e-5
res_fine = run_pipeline(im_fine, voxel_fine, shape_fine)
# run coarse (voxel doubled), physical size same
voxel_coarse = voxel_fine * factor[0]
res_coarse = run_pipeline(im_coarse.astype(int), voxel_coarse, shape_coarse)

print('Fine grid:', shape_fine, 'voxel=', voxel_fine)
print(res_fine)
print('Coarse grid:', shape_coarse, 'voxel=', voxel_coarse)
print(res_coarse)

# save to CSV
df = pd.DataFrame([{'grid':'fine', **res_fine}, {'grid':'coarse', **res_coarse}])
df.to_csv('voxel_scaling_test.csv', index=False, float_format='%.6e')
print('Saved voxel_scaling_test.csv')

# cleanup OpenPNM
try:
    op.workspace.clear()
except Exception:
    pass

print('Done')
