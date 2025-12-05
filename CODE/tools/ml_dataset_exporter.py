"""ML dataset exporter utilities

This module provides two capabilities:

- `save_sample_npz(out_dir, sample_id, seed, L, arrays, metadata)`:
    Called from inside a run to write a per-sample `.npz` containing arrays
    and metadata. Arrays is a dict mapping keys like `pore.coords`,
    `pore.diameter`, `throat.diameter`, `throat.length`, `pore.concentration`.

- `repackage_from_csv --csv <results.csv> --network-col <network_npz>`:
    CLI mode: read a results CSV that has a column pointing to per-run
    network snapshot `.npz` files (created during simulation). For each row
    that has a valid snapshot, create a normalized per-sample `.npz` into
    `DATASETS/<density>/` with a standard filename.

The exporter intentionally does not attempt to reconstruct networks from
OpenPNM pickles. It expects either that you saved a lightweight `.npz`
snapshot per-run during simulation, or that you have an external mapping
from CSV rows to snapshot files.
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, Any


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def save_sample_npz(out_dir: str, sample_id: int, seed: int, L: float, arrays: Dict[str, np.ndarray], metadata: Dict[str, Any]):
    """Save per-sample arrays + metadata into a single .npz file.

    - `out_dir` will be created if missing
    - `arrays` keys should be short strings (no slashes) describing the
      array (e.g., 'pore_coords', 'pore_diameter', 'throat_diameter', 'throat_length', 'concentration')
    - `metadata` will be saved as JSON under the key `_metadata.json` next to the .npz
    """
    ensure_dir(out_dir)
    fname = f"sample_{sample_id:06d}_seed{seed}_L{L:.6g}.npz"
    out_path = os.path.join(out_dir, fname)
    # Convert arrays dict keys to safe names
    npz_dict = {}
    for k, v in arrays.items():
        safe_k = k.replace('.', '_').replace('/', '_')
        npz_dict[safe_k] = np.asarray(v)

    # Save arrays
    np.savez_compressed(out_path, **npz_dict)

    # Save metadata JSON alongside
    meta_path = out_path + '.meta.json'
    meta = dict(metadata)
    meta.update({'sample_id': int(sample_id), 'seed': int(seed), 'L': float(L), 'created_utc': datetime.utcnow().isoformat()})
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print('Wrote', out_path)
    return out_path, meta_path


def repackage_from_csv(csv_path: str, network_col: str, out_base: str = 'DATASETS'):
    """Repackage existing per-run network snapshots referenced in a CSV.

    The CSV must contain at least a `seed` column and the column named by
    `network_col` which should point to an existing lightweight `.npz` or
    `.npz`-like file created during simulation. Each referenced file will be
    copied/repacked into `out_base/<density>/` with a normalized filename.
    Returns list of created files.
    """
    import csv
    created = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        # Try to pick density from CSV header rows if present in first row
        first = None
        rows = list(reader)
        if not rows:
            print('CSV empty:', csv_path)
            return created
        # Determine density if present
        density = rows[0].get('density', rows[0].get('density', 'unknown'))
        dlabel = str(density).replace('.', '_')
        target_dir = os.path.join(out_base, f'const_density_{dlabel}')
        ensure_dir(target_dir)

        for i, r in enumerate(rows, start=1):
            network_path = r.get(network_col, '')
            try:
                seed = int(r.get('seed', -1))
            except Exception:
                seed = -1
            try:
                L = float(r.get('L_m', r.get('L', 'nan')))
            except Exception:
                L = float('nan')

            if not network_path:
                print(f'Row {i}: no {network_col} entry, skipping')
                continue
            if not os.path.exists(network_path):
                print(f'Row {i}: referenced network file not found: {network_path}, skipping')
                continue

            # If it's an npz snapshot, load arrays and call save_sample_npz
            if network_path.lower().endswith('.npz'):
                try:
                    data = dict(np.load(network_path, allow_pickle=True))
                    # Map keys back to desired names if possible
                    arrays = {}
                    # Heuristic: accept keys like 'pore_coords' or 'pore.coords'
                    for k in data.keys():
                        arrays[k] = data[k]
                    sample_id = int(r.get('sample_id', i))
                    metadata = {k: r.get(k, None) for k in r.keys()}
                    out_path, meta_path = save_sample_npz(target_dir, sample_id, seed, L, arrays, metadata)
                    created.append(out_path)
                except Exception as e:
                    print(f'Failed to repackage {network_path}: {e}')
                    continue
            else:
                print(f'Unsupported snapshot format (need .npz): {network_path}, skipping')

    return created


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='Results CSV with a column pointing to per-run network snapshot (.npz)')
    p.add_argument('--network-col', default='network_npz', help='CSV column that contains path to per-run snapshot (.npz)')
    p.add_argument('--out-base', default='DATASETS', help='Base output folder')
    args = p.parse_args()

    created = repackage_from_csv(args.csv, args.network_col, args.out_base)
    print('Created', len(created), 'dataset samples in', args.out_base)


if __name__ == '__main__':
    main()
