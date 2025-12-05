"""Cleanup and organize CODE and FINAL SET directories.

This script performs safe, non-destructive reorganization:
- Renames the completed constant-density sweep folder (with timestamps) to
  `FINAL SET/const_density_1e13/` and normalizes the CSV and plot filenames.
- Moves files that look like backups, copies, or tests into `archive_unused/`
  with subfolders `copies/`, `backups/`, and `tests/`.

Run this from the project root. It prints actions and does not delete files.
"""

import os
import shutil
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_SET = ROOT / 'FINAL SET'
ARCHIVE = ROOT / 'archive_unused'


def mkdir(p):
    p.mkdir(parents=True, exist_ok=True)


def move_folder(src: Path, dst: Path):
    print(f"Moving folder: {src} -> {dst}")
    mkdir(dst.parent)
    try:
        shutil.move(str(src), str(dst))
    except Exception as e:
        print(f"Failed to move {src}: {e}")


def normalize_const_density():
    # Find folder matching sweep_rev_const_density_results_*_final_*
    pattern = re.compile(r'sweep_rev_const_density_results_.*_final_.*')
    matches = [p for p in FINAL_SET.iterdir() if p.is_dir() and pattern.match(p.name)]
    if not matches:
        print('No timestamped const-density folder found in FINAL SET.')
        return
    # If multiple matches, pick the most recent by mtime
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    src = matches[0]
    dst = FINAL_SET / 'const_density_1e13'
    if dst.exists():
        dst = FINAL_SET / 'const_density_1e13_1'
    shutil.move(str(src), str(dst))
    print(f"Renamed {src.name} -> {dst.name}")

    # Inside, rename the main CSV to a clear name if present
    csv_candidates = list(dst.glob('*.csv'))
    for c in csv_candidates:
        if 'const_density' in c.name or 'sweep' in c.name:
            new_name = dst / 'sweep_const_density_1e13.csv'
            print(f"Renaming CSV {c.name} -> {new_name.name}")
            shutil.move(str(c), str(new_name))
            break

    # Normalize rev_analysis plots
    for p in dst.glob('*rev_analysis*.png'):
        if 'mean_sem' in p.name:
            new = dst / 'rev_mean_sem.png'
        elif 'rel_diff' in p.name:
            new = dst / 'rev_rel_diff.png'
        elif 'cov' in p.name:
            new = dst / 'rev_cov.png'
        else:
            new = dst / p.name
        if new.exists():
            new = dst / (new.stem + '_1' + new.suffix)
        print(f"Renaming plot {p.name} -> {new.name}")
        shutil.move(str(p), str(new))


def archive_obvious_files():
    mkdir(ARCHIVE)
    copies = ARCHIVE / 'copies'
    backups = ARCHIVE / 'backups'
    tests = ARCHIVE / 'tests'
    others = ARCHIVE / 'others'
    mkdir(copies); mkdir(backups); mkdir(tests); mkdir(others)

    for p in ROOT.iterdir():
        if p.is_dir():
            # skip FINAL SET and tools and created archive
            if p.name in ('FINAL SET', 'tools', 'archive_unused'):
                continue
        if p.is_file():
            name = p.name.lower()
            try:
                if 'copy' in name:
                    print(f"Archiving copy: {p.name} -> archive/copies")
                    shutil.move(str(p), str(copies / p.name))
                elif 'backup' in name:
                    print(f"Archiving backup: {p.name} -> archive/backups")
                    shutil.move(str(p), str(backups / p.name))
                elif name.startswith('test') or name.startswith('test_') or name.startswith('test') or name.startswith('test') or name.startswith('tes') or name.startswith('temp') or name.startswith('debug'):
                    print(f"Archiving test/temp: {p.name} -> archive/tests")
                    shutil.move(str(p), str(tests / p.name))
                elif name.endswith('.png') and 'plot' in name and 'rev' in name:
                    # move general rev plots into FINAL SET/const_density_1e13 if present
                    target = FINAL_SET / 'const_density_1e13'
                    if target.exists():
                        print(f"Moving plot {p.name} -> {target.name}")
                        shutil.move(str(p), str(target / p.name))
                    else:
                        shutil.move(str(p), str(others / p.name))
                else:
                    # leave other files in place
                    pass
            except Exception as e:
                print(f"Failed moving {p}: {e}")


def mark_unused_folders():
    # Move obvious test folders into archive/tests
    test_folder_names = [n for n in os.listdir(FINAL_SET) if n.lower().startswith('test')]
    for name in test_folder_names:
        src = FINAL_SET / name
        dst = ARCHIVE / 'tests' / name
        if src.exists() and src.is_dir():
            print(f"Archiving FINAL SET test folder: {src} -> {dst}")
            mkdir(dst.parent)
            shutil.move(str(src), str(dst))


def main():
    print('Starting cleanup and organization...')
    normalize_const_density()
    archive_obvious_files()
    mark_unused_folders()
    print('Done. Review the changes under FINAL SET/ and archive_unused/.')


if __name__ == '__main__':
    main()
