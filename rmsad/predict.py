import pathlib
import sys
import pandas as pd

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from RMSAD_tool import get_RMSAD, get_shear_modulus_from_chemform  # noqa: E402

from rmsad.grid import generate_grid, ALL_ELEMENTS

_YS_CONSTANT = 0.29  # Å/eV — universal constant, Table 1 of Tandoc et al. 2023


def row_to_chemform(row: pd.Series) -> str:
    parts = []
    for el in ALL_ELEMENTS:
        val = float(row[el])
        if val > 0:
            parts.append(f"{el}{val:.6f}")
    return "".join(parts)


def _find_gamma_usf_col(df: pd.DataFrame) -> str | None:
    """Return the gamma_usf column name from a d-parameter HTP dataframe."""
    for col in df.columns:
        low = col.lower()
        if "usf" in low or "gsf" in low or ("gamma" in low and "usf" in low):
            return col
    for col in df.columns:
        low = col.lower()
        if "gsfe" in low or "usfe" in low or "gamma" in low:
            return col
    return None


def _merge_htp(grid_df: pd.DataFrame, system: str, dparameter_dir: pathlib.Path) -> pd.DataFrame:
    """Merge gamma_usf from the d-parameter HTP CSV and compute YS_GPa."""
    htp_path = dparameter_dir / f"predict_{system}_HTP.csv"
    if not htp_path.exists():
        print(f"  [warn] HTP file not found: {htp_path} — skipping YS")
        return grid_df

    htp = pd.read_csv(htp_path)
    gamma_col = _find_gamma_usf_col(htp)
    if gamma_col is None:
        print(f"  [warn] No gamma_usf column found in {htp_path.name} — skipping YS")
        print(f"         Columns present: {list(htp.columns)}")
        return grid_df

    # Merge on element fraction columns (round to avoid float drift)
    merge_cols = [el for el in ALL_ELEMENTS if el in htp.columns]
    for col in merge_cols:
        htp[col] = htp[col].round(6)
        grid_df[col] = grid_df[col].round(6)

    merged = grid_df.merge(htp[merge_cols + [gamma_col]], on=merge_cols, how="left")
    merged = merged.rename(columns={gamma_col: "gamma_usf"})
    merged["YS_GPa"] = _YS_CONSTANT * merged["mu_GPa"] * merged["gamma_usf"] * merged["RMSAD"]
    # Negative mu_GPa is non-physical; null out YS for those rows
    merged.loc[merged["mu_GPa"] <= 0, "YS_GPa"] = float("nan")

    missing = merged["gamma_usf"].isna().sum()
    if missing:
        print(f"  [warn] {missing} compositions had no gamma_usf match — YS will be NaN for those rows")

    print(f"  gamma_usf merged from {htp_path.name}  (column: '{gamma_col}')")
    return merged


def predict_system(
    system: str,
    output_dir: str | pathlib.Path = "data/output",
    step: int = 1,
    dparameter_dir: str | pathlib.Path | None = None,
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

    # Merge gamma_usf and compute YS if d-parameter output is available
    if dparameter_dir is not None:
        grid_df = _merge_htp(grid_df, system, pathlib.Path(dparameter_dir))

    suffix = f"_step{step}" if step != 1 else ""
    out_path = output_dir / f"predict_{system}_RMSAD{suffix}.csv"
    grid_df.to_csv(out_path, index=False)

    print(f"\nDone. {total} compositions predicted.")
    print(f"RMSAD  — min: {grid_df['RMSAD'].min():.4f}  mean: {grid_df['RMSAD'].mean():.4f}  max: {grid_df['RMSAD'].max():.4f}  Å")
    print(f"mu_GPa — min: {grid_df['mu_GPa'].min():.1f}  mean: {grid_df['mu_GPa'].mean():.1f}  max: {grid_df['mu_GPa'].max():.1f}  GPa")
    if "YS_GPa" in grid_df.columns:
        ys = grid_df["YS_GPa"].dropna()
        print(f"YS_GPa — min: {ys.min():.4f}  mean: {ys.mean():.4f}  max: {ys.max():.4f}  GPa")
    print(f"Output: {out_path}")

    return grid_df
