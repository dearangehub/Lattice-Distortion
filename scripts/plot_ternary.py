#!/usr/bin/env python3
"""
Usage:
  python scripts/plot_ternary.py TiNbV
  python scripts/plot_ternary.py TiNbV TiNbMo
  python scripts/plot_ternary.py TiNbV --ys-contour 900
  python scripts/plot_ternary.py TiNbV --contour

Plots YS_GPa on a ternary diagram with a labelled contour line at --ys-contour MPa.
Requires predict_{system}_RMSAD.csv to contain a YS_GPa column
(run run_prediction.py with --dparameter-dir first).

el1 = apex (top), el2 = bottom-left, el3 = bottom-right
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


def _ternary_to_xy(a_frac: np.ndarray, b_frac: np.ndarray, c_frac: np.ndarray):
    total = a_frac + b_frac + c_frac
    x = 0.5 * (a_frac + 2 * c_frac) / total
    y = (math.sqrt(3) / 2) * a_frac / total
    return x, y


def _draw_frame(ax, elements):
    el1, el2, el3 = elements
    sqrt3_2 = math.sqrt(3) / 2

    v_top = np.array([0.5, sqrt3_2])
    v_bl  = np.array([0.0, 0.0])
    v_br  = np.array([1.0, 0.0])

    tick_len = 0.02
    for i in range(1, 10):
        t = i / 10.0
        label = f"{int(t * 100)}%"

        p = v_bl + t * (v_br - v_bl)
        ax.plot([p[0], p[0]], [p[1], p[1] - tick_len], "k-", lw=0.5)
        ax.text(p[0], p[1] - 0.04, label, ha="center", va="top", fontsize=6)

        p = v_top + t * (v_bl - v_top)
        edge = v_bl - v_top
        perp = np.array([-edge[1], edge[0]]) / np.linalg.norm([-edge[1], edge[0]])
        ax.plot([p[0], p[0] + tick_len * perp[0]], [p[1], p[1] + tick_len * perp[1]], "k-", lw=0.5)
        ax.text(p[0] - 0.04, p[1], label, ha="right", va="center", fontsize=6)

        p = v_top + t * (v_br - v_top)
        edge = v_br - v_top
        perp = np.array([edge[1], -edge[0]]) / np.linalg.norm([edge[1], -edge[0]])
        ax.plot([p[0], p[0] + tick_len * perp[0]], [p[1], p[1] + tick_len * perp[1]], "k-", lw=0.5)
        ax.text(p[0] + 0.04, p[1], label, ha="left", va="center", fontsize=6)

    ax.add_patch(plt.Polygon([v_top, v_bl, v_br], fill=False, edgecolor="black", lw=1))

    mid_bottom = (v_bl + v_br) / 2
    ax.text(mid_bottom[0], mid_bottom[1] - 0.10, f"{el2} at.%", ha="center", va="top", fontsize=9)

    mid_left = (v_top + v_bl) / 2
    ax.text(mid_left[0] - 0.16, mid_left[1], f"{el1} at.%", ha="center", va="center",
            fontsize=9, rotation=math.degrees(math.atan2(*(v_bl - v_top)[::-1])))

    mid_right = (v_top + v_br) / 2
    ax.text(mid_right[0] + 0.16, mid_right[1], f"{el3} at.%", ha="center", va="center",
            fontsize=9, rotation=math.degrees(math.atan2(*(v_br - v_top)[::-1])))

    ax.text(v_top[0], v_top[1] + 0.04, el1, ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(v_bl[0] - 0.04, v_bl[1] - 0.04, el2, ha="right", va="top", fontsize=11, fontweight="bold")
    ax.text(v_br[0] + 0.04, v_br[1] - 0.04, el3, ha="left", va="top", fontsize=11, fontweight="bold")


def plot_system(system: str, ax, contour_fill: bool, ys_contour_mpa: float,
                vmin=None, vmax=None):
    csv_path = OUTPUT_DIR / f"predict_{system}_RMSAD.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No prediction file: {csv_path}\n"
            f"Run: python scripts/run_prediction.py {system} --dparameter-dir <dir>"
        )

    df = pd.read_csv(csv_path)
    if "YS_GPa" not in df.columns:
        raise ValueError(
            f"YS_GPa column not found in {csv_path.name}.\n"
            f"Re-run: python scripts/run_prediction.py {system} --dparameter-dir <dir>"
        )

    elements = _parse_elements(system)
    el1, el2, el3 = elements

    a = df[el1].values
    b = df[el2].values
    c = df[el3].values
    z = df["YS_GPa"].values * 1000  # convert GPa -> MPa for display

    mask = ~np.isnan(z)
    a, b, c, z = a[mask], b[mask], c[mask], z[mask]

    x, y = _ternary_to_xy(a, b, c)
    triang = mtri.Triangulation(x, y)

    if contour_fill:
        sc = ax.tricontourf(triang, z, levels=20, cmap="YlOrRd", vmin=vmin, vmax=vmax)
    else:
        sc = ax.tripcolor(triang, z, cmap="YlOrRd", shading="gouraud", vmin=vmin, vmax=vmax)

    # Labelled contour line at the requested YS threshold (already in MPa)
    if ys_contour_mpa is not None and vmin <= ys_contour_mpa <= vmax:
        cs = ax.tricontour(triang, z, levels=[ys_contour_mpa], colors=["black"], linewidths=1.5)
        ax.clabel(cs, fmt=f"{int(ys_contour_mpa)} MPa", fontsize=8, inline=True)

    _draw_frame(ax, elements)
    ax.set_xlim(-0.20, 1.20)
    ax.set_ylim(-0.15, math.sqrt(3) / 2 + 0.10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(system, fontsize=12, pad=10)
    return sc


def main():
    parser = argparse.ArgumentParser(description="Plot YS ternary diagram for RHEA systems")
    parser.add_argument("systems", nargs="+", help="System names, e.g. TiNbV TiNbMo")
    parser.add_argument(
        "--ys-contour",
        type=float,
        default=900,
        metavar="MPa",
        help="Draw a labelled contour line at this YS value in MPa (default: 900)",
    )
    parser.add_argument(
        "--contour",
        action="store_true",
        help="Use contour fill instead of smooth tripcolor",
    )
    args = parser.parse_args()

    ys_contour_mpa = args.ys_contour  # already in MPa

    n = len(args.systems)
    fig = plt.figure(figsize=(5 * n + 1, 5))
    gs = gridspec.GridSpec(1, n + 1, width_ratios=[5] * n + [0.3], figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cax  = fig.add_subplot(gs[0, n])

    # Shared colour range across all systems (in MPa)
    all_vals = []
    for system in args.systems:
        csv_path = OUTPUT_DIR / f"predict_{system}_RMSAD.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "YS_GPa" in df.columns:
                all_vals.extend((df["YS_GPa"].dropna().values * 1000).tolist())
    vmin = min(all_vals) if all_vals else 0
    vmax = max(all_vals) if all_vals else 1

    sc = None
    for ax, system in zip(axes, args.systems):
        sc = plot_system(system, ax, args.contour, ys_contour_mpa, vmin=vmin, vmax=vmax)

    if sc is not None:
        fig.colorbar(sc, cax=cax, label="Yield Strength (MPa)")

    fig.suptitle("Yield Strength (YS) — Ternary Prediction", fontsize=13, y=1.01)
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"ternary_YS_{'_'.join(args.systems)}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
