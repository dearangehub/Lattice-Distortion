import pathlib
import pandas as pd

ALL_ELEMENTS = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Mo", "W", "Re", "Ru"]

# Two-char elements must be checked before one-char to avoid mis-splitting
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


def generate_ternary_grid(system: str, output_dir: str | pathlib.Path = "data/output") -> pd.DataFrame:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    elements = _parse_elements(system)
    if len(elements) != 3:
        raise ValueError(f"Expected 3 elements, got {len(elements)}: {elements}")

    el1, el2, el3 = elements

    rows = []
    idx = 0
    for a in range(1, 99):
        for b in range(1, 100 - a):
            c = 100 - a - b
            row = {el: 0.0 for el in ALL_ELEMENTS}
            row[el1] = round(a / 100, 6)
            row[el2] = round(b / 100, 6)
            row[el3] = round(c / 100, 6)
            sample = f"{el1}{a}{el2}{b}{el3}{c}"
            rows.append({"index": idx, "sample": sample, **{el: row[el] for el in ALL_ELEMENTS}})
            idx += 1

    df = pd.DataFrame(rows, columns=["index", "sample"] + ALL_ELEMENTS)
    out_path = output_dir / f"grid_{system}.csv"
    df.to_csv(out_path, index=False)
    print(f"Grid saved to {out_path}  ({len(df)} compositions)")
    return df


def get_grid_csv(system: str, output_dir: str | pathlib.Path = "data/output") -> pathlib.Path:
    output_dir = pathlib.Path(output_dir)
    path = output_dir / f"grid_{system}.csv"
    if not path.exists():
        generate_ternary_grid(system, output_dir)
    return path
