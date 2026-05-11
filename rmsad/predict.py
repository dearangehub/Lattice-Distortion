import pathlib
import sys
import pandas as pd

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from RMSAD_tool import get_RMSAD, get_shear_modulus_from_chemform  # noqa: E402

from rmsad.grid import generate_grid, ALL_ELEMENTS


def row_to_chemform(row: pd.Series) -> str:
    parts = []
    for el in ALL_ELEMENTS:
        val = float(row[el])
        if val > 0:
            parts.append(f"{el}{val:.6f}")
    return "".join(parts)


def predict_system(
    system: str,
    output_dir: str | pathlib.Path = "data/output",
    step: int = 1,
) -> pd.DataFrame:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_df = generate_grid(system, output_dir, step)

    rmsad_vals = []
    mu_vals = []
    total = len(grid_df)

    print(f"Predicting RMSAD and shear modulus for {system} ({total} compositions)...")

    for i, row in grid_df.iterrows():
        chemform = row_to_chemform(row)
        rmsad_vals.append(get_RMSAD(chemform))
        mu_vals.append(get_shear_modulus_from_chemform(chemform))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{total} compositions processed...")

    grid_df["RMSAD"] = rmsad_vals
    grid_df["mu_GPa"] = mu_vals

    suffix = f"_step{step}" if step != 1 else ""
    out_path = output_dir / f"predict_{system}_RMSAD{suffix}.csv"
    grid_df.to_csv(out_path, index=False)

    print(f"\nDone. {total} compositions predicted.")
    print(f"RMSAD  — min: {grid_df['RMSAD'].min():.4f}  mean: {grid_df['RMSAD'].mean():.4f}  max: {grid_df['RMSAD'].max():.4f}  Å")
    print(f"mu_GPa — min: {grid_df['mu_GPa'].min():.1f}  mean: {grid_df['mu_GPa'].mean():.1f}  max: {grid_df['mu_GPa'].max():.1f}  GPa")
    print(f"Output: {out_path}")

    return grid_df
