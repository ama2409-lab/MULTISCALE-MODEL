Project To-Do (short tracker)

Priority actionable items:

- Implement new-density sweep driver (`tools/run_density_sweep.py`) — completed.
- Create sweep manifest & README for completed sweep (`FINAL SET/const_density_1e13/manifest.json`, `README.md`) — in-progress.
- Export ML dataset samples into `DATASETS/` (decide `.npz` vs per-sample folders) — pending.
- Add `requirements.txt` and brief environment notes (headless runs, optional packages) — pending.
- Provide PowerShell sleep-inhibitor helper and Scheduled Task example to resume sweeps after reboot — pending.
- Run a pilot new-density sweep (e.g., `--density 1e12`) with reduced sizes/seeds — completed.

Use `tools/run_density_sweep.py --density <value> --sizes <L1,L2,..> --seeds <start-end>` to run a new sweep.
