# Sweep results — const_density_1e13

This folder contains the completed fixed-density REV sweep for `density = 1e13 pores/m^3`.

Summary
- Density: 1e13 pores/m^3
- Seeds used: 100–107
- Results CSV: `sweep_const_density_1e13.csv` (per-run rows with Dirichlet/Neumann D_eff, porosity, tau, geometry repair counts)

Key files
- `sweep_const_density_1e13.csv` — primary results table (timestamped rows)
- `rev_mean_sem.png`, `rev_rel_diff.png`, `rev_cov.png` — analysis plots used to detect candidate REV
- `rev_stats_20251120_110857.csv`, `rev_report_20251120_110857.txt` — detailed diagnostics and bootstrap summaries
- `rev_diag_20251120_110857_dir_box.png`, `rev_diag_20251120_110857_neu_box.png`, `rev_diag_20251120_110857_violin.png` — distribution plots across seeds

How this was produced
- Networks were generated with `FINAL SET/NETWORK_GENERATION_FINAL.py` at fixed pore density.
- The driver `FINAL SET/sweep_rev_sizes.py` was used to generate networks, apply conservative geometry repairs, compute robust conductances and run Dirichlet and Neumann Fickian diffusion; results were appended to the CSV.

Reproducibility/provenance
- See `manifest.json` (this folder) for a short manifest listing density, seeds and file inventory.
- The full per-run metadata (L, seed, Np, Nt, geometry repair counts, D_eff values, porosity, etc.) is contained in the CSV.

Next steps (suggested)
- Create dataset exporter to write per-sample arrays (`pore.coords`, `pore.diameter`, `throat.*`, `pore.concentration`) into `DATASETS/` for ML training.
- Add `requirements.txt` and a headless-run environment note.
- Implement `tools/run_density_sweep.py` (driver) to run sweeps at arbitrary densities (a driver was added under `tools/`).
