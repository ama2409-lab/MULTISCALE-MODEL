Dataset generator README

This repository contains scripts to generate porous microstructure images (porespy), extract pore networks (snow2 -> OpenPNM), run Fickian diffusion simulations, and save per-sample descriptors and results to CSV for ML.

Files of interest
- `mainDataSetGenerator.py` - Batch dataset generator using Latin Hypercube Sampling (LHS). Produces a CSV with descriptors and diffusion results.
- `test_voxel_scaling.py` - Small test comparing D_eff and descriptors for different voxel sizes (keeps physical size fixed).
- `test_diffusivity_norm.py` - Small test verifying D_eff scales with molecular diffusivity and that D_eff_norm = D_eff/D_AB is dimensionless.

CSV format produced by `mainDataSetGenerator.py`
- sample_id: integer sample index (1-based)
- porosity_requested: porosity requested to the porespy generator (unitless fraction)
- blobiness: blobiness parameter passed to porespy (unitless)
- seed: per-sample RNG seed used to make the sample reproducible (integer)
- voxel: voxel size used for that sample in meters (float)
- shape: image shape as a string tuple, e.g. "(30, 30, 15)" (useful to infer Nx,Ny,Nz)
- error: traceback string if the sample failed; empty if successful

Geometric descriptors (per-sample)
- corr_length_vox: correlation length from radial autocorrelation in voxels (float)
- corr_length_m: correlation length in meters (float)
- euler_char: Euler characteristic of the binary pore space (integer or float)
- pore_diam_count: number of detected local maxima in distance transform (int)
- pore_diam_mean_m: mean pore diameter estimated from distance transform (meters)
- pore_diam_median_m: median pore diameter (meters)
- pore_diam_p10_m: 10th percentile diameter (meters)
- pore_diam_p90_m: 90th percentile diameter (meters)

Network and simulation results
- Np: number of pores in extracted OpenPNM network (int)
- Nt: number of throats (int)
- im_porosity: measured porosity fraction from the binary image (unitless)
- molar_flow_rate: total molar flow rate at inlet (mol/s)
- D_eff: computed effective diffusivity (m^2/s)
- D_eff_norm: D_eff divided by molecular diffusivity assigned to the phase (dimensionless)
- porosity_net: porosity computed from pore+throat volumes (unitless)
- tortuosity: tortuosity estimate tau = e * D_AB / D_eff (dimensionless)
- V_p: total pore volume (m^3)
- V_t: total throat volume (m^3)
- V_bulk: total bulk volume (m^3)
- runtime_s: runtime in seconds for the sample
- image_file, network_file: (optional) file paths to saved image slices and network snapshot if `--save-images` is used

Recommended preprocessing for ML
1. Filtering
   - Remove rows where `error` is not empty. Those samples failed and contain traceback strings.
   - Optionally filter samples where `Np` or `pore_diam_count` are extremely small; they may be degenerate microstructures.

2. Unit handling and normalization
   - `D_eff` is in SI units (m^2/s). Use `D_eff_norm` as a dimensionless target to remove dependence on absolute molecular diffusivity if you want microstructure-only prediction.
   - Keep `voxel` and `shape` columns so you can control or normalize geometric features that depend on sampling resolution.

3. Feature engineering suggestions
   - Use `corr_length_vox` and `corr_length_m` as features; both are useful (voxels encode sampling resolution; meters encode length scale).
   - Use summary statistics of pore diameters (`pore_diam_mean_m`, `pore_diam_median_m`, `pore_diam_p10_m`, `pore_diam_p90_m`). Consider log-transforming diameters.
   - Use `im_porosity` and `porosity_net` — they differ slightly; include both or prefer `porosity_net` which is from volumes.
   - Normalize size-dependent features by a physical length scale (e.g., correlation length or system L) if you want ML invariance to voxel or system size.

4. Train/validation split
   - Split by LHS samples or use stratified sampling over porosity/blobiness to ensure coverage across design space.
   - Keep an out-of-distribution held-out set by holding out the highest/lowest porosity range for testing generalization.

Quick commands
Run a 10-sample batch (example):

```powershell
python mainDataSetGenerator.py --samples 10 --shape 30 30 15 --out mydataset.csv --seed 42
```

Run the voxel-scaling test:

```powershell
python test_voxel_scaling.py
```

If you want help
- I can add a small helper that flattens `shape` into `Nx,Ny,Nz` columns, or create per-sample folders for images.
- I can also add a small notebook that demonstrates common preprocessing steps (filtering, scaling, train/test split) and trains a baseline regression model for D_eff_norm.

License: MIT (use as you wish)
