"""
rmsad/predict.py — Batch RMSAD and shear modulus prediction.

For each composition row in a grid CSV, computes:
  - RMSAD (Å)
  - mu_GPa (isotropic shear modulus via VRH + Vegard's law)

YS prediction requires gamma_usf from the d-parameter pipeline.
Merge the two output CSVs on composition columns to compute YS.
"""

import sys
import os
import pandas as pd
from pathlib import Path


def _get_repo_root():
    """Auto-detect repo root as two levels up from this file."""
    return Path(__file__).resolve().parent.parent


def row_to_chemform(row):
    """
    Build a composition string from a CSV row.
    Only includes elements with non-zero composition.
    Format: "Ti0.330000Nb0.340000V0.330000"
    Fixed element order: Ti, Zr, Hf, V, Nb, Ta, Mo, W, Re, Ru
    """
    elements = ['Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Mo', 'W', 'Re', 'Ru']
    return ''.join(f'{el}{row[el]:.6f}' for el in elements if row[el] > 0)


def predict_system(system, repo_root=None, output_dir=None):
    """
    Run batch RMSAD + shear modulus prediction for a ternary system.

    Parameters
    ----------
    system     : str   e.g. "TiNbV"
    repo_root  : Path  repo root (auto-detected if None)
    output_dir : Path  output directory (defaults to repo_root/data/output)

    Returns
    -------
    Path to output CSV
    """
    if repo_root is None:
        repo_root = _get_repo_root()
    repo_root = Path(repo_root)

    if output_dir is None:
        output_dir = repo_root / 'data' / 'output'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add repo root to sys.path so RMSAD_tool.py is importable
    sys.path.insert(0, str(repo_root))
    from RMSAD_tool import get_RMSAD, get_shear_modulus_from_chemform

    # Get or generate composition grid CSV
    from rmsad.grid import get_grid_csv
    grid_path = get_grid_csv(system, repo_root)
    df = pd.read_csv(grid_path)

    print(f"Predicting RMSAD and shear modulus for {system} "
          f"({len(df)} compositions)...")

    rmsad_vals = []
    mu_vals = []

    for i, row in df.iterrows():
        chemform = row_to_chemform(row)
        rmsad_vals.append(get_RMSAD(chemform))
        mu_vals.append(get_shear_modulus_from_chemform(chemform))

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(df)} done...")

    df['RMSAD'] = rmsad_vals
    df['mu_GPa'] = mu_vals

    out_path = output_dir / f'predict_{system}_RMSAD.csv'
    df.to_csv(out_path, index=False)

    print(f"\nDone. {len(df)} compositions predicted.")
    print(f"RMSAD  — min: {df['RMSAD'].min():.4f}  "
          f"mean: {df['RMSAD'].mean():.4f}  "
          f"max: {df['RMSAD'].max():.4f}  Å")
    print(f"mu_GPa — min: {df['mu_GPa'].min():.1f}  "
          f"mean: {df['mu_GPa'].mean():.1f}  "
          f"max: {df['mu_GPa'].max():.1f}  GPa")
    print(f"Output: {out_path}")

    return out_path
