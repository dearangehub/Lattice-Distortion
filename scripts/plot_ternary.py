#!/usr/bin/env python3
"""
Usage:
  python scripts/plot_ternary.py TiNbV
  python scripts/plot_ternary.py TiNbV TiNbMo
  python scripts/plot_ternary.py TiNbV --column RMSAD
  python scripts/plot_ternary.py TiNbV --contour

el1 = apex (top), el2 = bottom-left, el3 = bottom-right
Column plotted: YS_GPa if present, otherwise RMSAD (override with --column)
"""
import argparse
import math
import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from rmsad.grid import _parse_elements  # noqa: E402

OUTPUT_DIR = repo_root / "data" / "output"

_COLUMN_META = {
    "YS_GPa":    {"label": "Yield Strength (GPa)", "cmap": "YlOrRd"},
    "RMSAD":     {"label": "RMSAD (Å)",            "cmap": "YlOrRd"},
    "mu_GPa":    {"label": "Shear Modulus (GPa)",  "cmap": "Blues"},
    "gamma_usf": {"label": "γ_USF (J/m²)",         "cmap": "Greens"},
}


def _ternary_to_xy(a_frac: np.ndarray, b_frac: np.ndarray, c_frac: np.ndarray):
    """el1=a (top), el2=b (bottom-left), el3=c (bottom-right)."""
    total = a_frac + b_frac + c_frac
    x = 0.5 * (a_frac + 2 * c_frac) / total
    y = (math.sqrt(3) / 2) * a_frac / total
    return x, y


def _draw_ticks_and_labels(ax, elements):
    el1, el2, el3 = elements
    sqrt3_2 = math.sqrt(3) / 2

    v_top = np.array([0.5, sqrt3_2])
    v_bl  = np.array([0.0, 0.0])
    v_br  = np.array([1.0, 0.0])

    tick_len = 0.02
    for i in range(1, 10):
        t = i / 10.0
        label = f"{int(t * 100)}%"

        # Bottom edge (el2 → el3)
        p = v_bl + t * (v_br - v_bl)
        ax.plot([p[0], p[0]], [p[1], p[1] - tick_len], "k-", lw=0.5)
        ax.text(p[0], p[1] - 0.04, label, ha="center", va="top", fontsize=6)

        # Left edge (el1 → el2)
        p = v_top + t * (v_bl - v_top)
        edge = v_bl - v_top
        perp = np.array([-edge[1], edge[0]])
        perp /= np.linalg.norm(perp)
        ax.plot([p[0], p[0] + tick_len * perp[0]], [p[1], p[1] + tick_len * perp[1]], "k-", lw=0.5)
        ax.text(p[0] - 0.04, p[1], label, ha="right", va="center", fontsize=6)

        # Right edge (el1 → el3)
        p = v_top + t * (v_br - v_top)
        edge = v_br - v_top
        perp = np.array([edge[1], -edge[0]])
        perp /= np.linalg.norm(perp)
        ax.plot([p[0], p[0] + tick_len * perp[0]], [p[1], p[1] + tick_len * perp[1]], "k-", lw=0.5)
        ax.text(p[0] + 0.04, p[1], label, ha="left", va="center", fontsize=6)

    triangle = plt.Polygon([v_top, v_bl, v_br], fill=False, edgecolor="black", lw=1)
    ax.add_patch(triangle)

    mid_bottom = (v_bl + v_br) / 2
    ax.text(mid_bottom[0], mid_bottom[1] - 0.10, f"{el2} at.%", ha="center", va="top", fontsize=9)

    mid_left = (v_top + v_bl) / 2
    ax.text(mid_left[0] - 0.08, mid_left[1], f"{el1} at.%", ha="center", va="center",
            fontsize=9, rotation=math.degrees(math.atan2(*(v_bl - v_top)[::-1])))

    mid_right = (v_top + v_br) / 2
    ax.text(mid_right[0] + 0.08, mid_right[1], f"{el3} at.%", ha="center", va="center",
            fontsize=9, rotation=math.degrees(math.atan2(*(v_br - v_top)[::-1])))

    ax.text(v_top[0], v_top[1] + 0.04, el1, ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(v_bl[0] - 0.04, v_bl[1] - 0.04, el2, ha="right", va="top", fontsize=11, fontweight="bold")
    ax.text(v_br[0] + 0.04, v_br[1] - 0.04, el3, ha="left", va="top", fontsize=11, fontweight="bold")


def _pick_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Column '{requested}' not in CSV. Available: {list(df.columns)}")
        return requested
    for col in ("YS_GPa", "RMSAD"):
        if col in df.columns:
            return col
    raise ValueError("No plottable column found (YS_GPa or RMSAD). Run prediction first.")


def plot_system(system: str, ax, column: str | None, contour: bool, vmin=None, vmax=None):
    csv_path = OUTPUT_DIR / f"predict_{system}_RMSAD.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No prediction file: {csv_path}\nRun: python scripts/run_prediction.py {system}")

    df = pd.read_csv(csv_path)
    col = _pick_column(df, column)
    meta = _COLUMN_META.get(col, {"label": col, "cmap": "YlOrRd"})

    elements = _parse_elements(system)
    el1, el2, el3 = elements

    a, b, c = df[el1].values, df[el2].values, df[el3].values
    z = df[col].values

    mask = ~np.isnan(z)
    a, b, c, z = a[mask], b[mask], c[mask], z[mask]

    x, y = _ternary_to_xy(a, b, c)
    triang = mtri.Triangulation(x, y)

    if contour:
        sc = ax.tricontourf(triang, z, levels=20, cmap=meta["cmap"], vmin=vmin, vmax=vmax)
    else:
        sc = ax.tripcolor(triang, z, cmap=meta["cmap"], shading="gouraud", vmin=vmin, vmax=vmax)

    _draw_ticks_and_labels(ax, elements)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, math.sqrt(3) / 2 + 0.10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(system, fontsize=12, pad=10)
    return sc, meta["label"]


def main():
    parser = argparse.ArgumentParser(description="Plot ternary composition diagrams")
    parser.add_argument("systems", nargs="+", help="System names, e.g. TiNbV TiNbMo")
    parser.add_argument("--column", default=None,
                        help="Column to plot (default: YS_GPa if available, else RMSAD). "
                             "Options: YS_GPa, RMSAD, mu_GPa, gamma_usf")
    parser.add_argument("--contour", action="store_true",
                        help="Use contour fill instead of smooth tripcolor")
    args = parser.parse_args()

    n = len(args.systems)
    fig = plt.figure(figsize=(5 * n + 1, 5))
    gs = gridspec.GridSpec(1, n + 1, width_ratios=[5] * n + [0.3], figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cax  = fig.add_subplot(gs[0, n])

    all_vals = []
    col_used = args.column
    for system in args.systems:
        csv_path = OUTPUT_DIR / f"predict_{system}_RMSAD.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            try:
                c = _pick_column(df, col_used)
                col_used = col_used or c
                all_vals.extend(df[c].dropna().values)
            except ValueError:
                pass
    vmin = min(all_vals) if all_vals else None
    vmax = max(all_vals) if all_vals else None

    sc, cb_label = None, col_used or "value"
    for ax, system in zip(axes, args.systems):
        sc, cb_label = plot_system(system, ax, col_used, args.contour, vmin=vmin, vmax=vmax)

    if sc is not None:
        fig.colorbar(sc, cax=cax, label=cb_label)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"ternary_{'_'.join(args.systems)}_{col_used or 'YS_GPa'}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
