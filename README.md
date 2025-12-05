# Multiscale Porous Media Transport – REV Sweeps

This repository contains the scripts and analysis used to study
representative elementary volume (REV) behaviour of diffusion
through Voronoi pore networks at multiple pore densities.

## Layout

- `CODE/` – Python scripts for network generation, sweeps, and analysis.
  - `FINAL SET/` – final sweep and analysis scripts.
  - `tools/` – helper utilities (sweep runners, dataset exporters, plotting).
- `DATASETS/` – generated datasets (ignored by Git; regenerated as needed).
- `Resources/` – figures or supplementary material for the report.

## Installation

```bash
python -m venv .venv
.venv\\Scripts\\activate  # on Windows

pip install -r requirements.txt
```

## Running REV Sweeps

Constant-density sweeps are driven by:

```bash
python CODE/tools/run_full_rev_1e12.py --density 1e12
python CODE/tools/run_full_rev_1e12.py --density 1e13
python CODE/tools/run_full_rev_1e12.py --density 1e14
```

Key options:

- `--target` – target realizations per `L_m` bin (default 100).
- `--seed-start` – starting random seed for new runs.
- `--chunk` – number of seeds per batch call.
- `--only` – comma-separated subset of `L_m` values to process.
- `--density` – pore density, e.g. `1e9`, `1e12`, `1e13`, `1e14`.

Outputs are written under `CODE/FINAL SET/const_density_<density_tag>/`
as `sweep_const_density_<density_tag>_full.csv`.

## Comparative Plots and REV Summary

Once sweeps are complete for your densities of interest, generate
harmonised comparison plots and REV plateau statistics with:

```bash
python -c "from CODE.tools.compare_density_sweeps import main; main()"
```

This creates:

- `CODE/FINAL SET/plots/diffusivity_vs_L.png`
- `CODE/FINAL SET/plots/tortuosity_vs_L.png`
- `CODE/FINAL SET/plots/porosity_vs_L.png`
- `CODE/FINAL SET/plots/rev_plateau_summary.csv`

You can drop these figures and the CSV directly into your slides or paper.

## Notes

- Large raw datasets and intermediate result folders are ignored via `.gitignore`
  to keep the repository light and reproducible.
- If you adjust the REV definition, edit `PLATEAU_TOLERANCE` in
  `CODE/tools/compare_density_sweeps.py` and rerun the plotting command.
