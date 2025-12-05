Headless run notes

Quick tips to run the scripts headless (no GUI) and reproducibly on Windows:

- Use the `Agg` backend for Matplotlib (many scripts already do this):
  - In Python: `import matplotlib; matplotlib.use('Agg')` before importing `pyplot`.

- Run long sweeps from PowerShell in a background process to detach from the interactive session:
  ```powershell
  Start-Process -FilePath python -ArgumentList 'tools/run_density_sweep.py --density 1e12 --sizes 1e-05,1e-04 --seeds 100-103' -NoNewWindow
  ```

- If you need to keep the laptop awake for long runs, consider temporarily disabling sleep via PowerShell (requires admin to change system wide settings):
  ```powershell
  # disable sleep on AC power (returns immediately)
  powercfg /change standby-timeout-ac 0
  # restore after run (example 30 minutes)
  powercfg /change standby-timeout-ac 30
  ```

- Avoid importing heavy optional packages during batch-startup: the driver `tools/run_density_sweep.py` batches seeds per-L to avoid repeated heavy imports.

- Run inside a Python virtual environment to control package versions. Create and activate venv (PowerShell):
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

If you want, I can add a small PowerShell helper script that:
- creates/activates a venv,
- applies temporary sleep inhibitor,
- launches the sweep in the background redirecting stdout/stderr to a logfile.
