import pathlib
import sys
import pandas as pd

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from RMSAD_tool import get_RMSAD  # noqa: E402

from rmsad.grid import generate_ternary_grid, ALL_ELEMENTS


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
) -> pd.DataFrame:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_df = generate_ternary_grid(system, output_dir)

    results = []
    total = len(grid_df)
    for i, row in grid_df.iterrows():
        chemform = row_to_chemform(row)
        rmsad_val = get_RMSAD(chemform)
        results.append(rmsad_val)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{total} compositions processed...")

    grid_df["RMSAD"] = results
    out_path = output_dir / f"predict_{system}_RMSAD.csv"
    grid_df.to_csv(out_path, index=False)

    rmsad_series = grid_df["RMSAD"]
    print(f"\n{system} RMSAD summary:")
    print(f"  min  = {rmsad_series.min():.6f} Å")
    print(f"  mean = {rmsad_series.mean():.6f} Å")
    print(f"  max  = {rmsad_series.max():.6f} Å")
    print(f"Results saved to {out_path}")

    return grid_df
