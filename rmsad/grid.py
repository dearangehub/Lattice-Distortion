import pathlib
from typing import Iterator
import pandas as pd

ALL_ELEMENTS = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Mo", "W", "Re", "Ru"]

_TWO_CHAR = {"Ti", "Zr", "Hf", "Nb", "Ta", "Mo", "Re", "Ru"}
_ONE_CHAR = {"V", "W"}


def _parse_elements(system: str) -> list[str]:
    if "-" in system:
        return system.split("-")
    elements = []
    i = 0
    while i < len(system):
        if i + 1 < len(system) and system[i : i + 2] in _TWO_CHAR:
            elements.append(system[i : i + 2])
            i += 2
        elif system[i] in _ONE_CHAR:
            elements.append(system[i])
            i += 1
        else:
            raise ValueError(f"Cannot parse element at position {i} in '{system}'")
    return elements


def _iter_compositions(n: int, total: int, step: int, min_val: int) -> Iterator[tuple[int, ...]]:
    """Yield all n-tuples of multiples of step that sum to total, each >= min_val."""
    if n == 1:
        if total >= min_val and total % step == 0:
            yield (total,)
        return
    for val in range(min_val, total - (n - 1) * min_val + 1, step):
        for rest in _iter_compositions(n - 1, total - val, step, min_val):
            yield (val,) + rest


def generate_grid(
    system: str,
    output_dir: str | pathlib.Path = "data/output",
    step: int = 1,
) -> pd.DataFrame:
    """Generate a composition grid for any n-component alloy system.

    Each element gets at least `step` at.% and all fractions are multiples of `step`.
    Typical step sizes: 1 (binary/ternary), 5 (quaternary), 10 (quinary+).
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    elements = _parse_elements(system)
    n = len(elements)
    if n < 2:
        raise ValueError(f"Need at least 2 elements, got {n}")
    for el in elements:
        if el not in ALL_ELEMENTS:
            raise ValueError(f"Unsupported element '{el}'. Supported: {ALL_ELEMENTS}")

    rows = []
    for idx, fracs in enumerate(_iter_compositions(n, 100, step, step)):
        row = {el: 0.0 for el in ALL_ELEMENTS}
        sample_parts = []
        for el, pct in zip(elements, fracs):
            row[el] = round(pct / 100, 6)
            sample_parts.append(f"{el}{pct}")
        rows.append({"index": idx, "sample": "".join(sample_parts), **{el: row[el] for el in ALL_ELEMENTS}})

    df = pd.DataFrame(rows, columns=["index", "sample"] + ALL_ELEMENTS)
    suffix = f"_step{step}" if step != 1 else ""
    out_path = output_dir / f"grid_{system}{suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"Grid saved to {out_path}  ({len(df)} compositions)")
    return df


# Backward-compatible alias
def generate_ternary_grid(system: str, output_dir: str | pathlib.Path = "data/output", step: int = 1) -> pd.DataFrame:
    return generate_grid(system, output_dir, step)


def get_grid_csv(system: str, output_dir: str | pathlib.Path = "data/output", step: int = 1) -> pathlib.Path:
    output_dir = pathlib.Path(output_dir)
    suffix = f"_step{step}" if step != 1 else ""
    path = output_dir / f"grid_{system}{suffix}.csv"
    if not path.exists():
        generate_grid(system, output_dir, step)
    return path
