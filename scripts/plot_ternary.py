#!/usr/bin/env python3
"""
Usage: python scripts/plot_ternary.py SYSTEM [SYSTEM ...] [--contour]

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
    """el1=a (top), el2=b (bottom-left), el3=c (bottom-right)."""
    x = 0.5 * (a_frac + 2 * c_frac) / (a_frac + b_frac + c_frac)
    y = (math.sqrt(3) / 2) * a_frac / (a_frac + b_frac + c_frac)
    return x, y


def _draw_ticks_and_labels(ax, elements):
    el1, el2, el3 = elements
    sqrt3_2 = math.sqrt(3) / 2

    # Triangle vertices: top, bottom-left, bottom-right
    v_top = np.array([0.5, sqrt3_2])
    v_bl = np.array([0.0, 0.0])
    v_br = np.array([1.0, 0.0])

    tick_len = 0.02
    for i in range(1, 10):
        t = i / 10.0

        # Bottom edge (el2 -> el3, a=0)
        p = v_bl + t * (v_br - v_bl)
        perp = np.array([0, -1])
        ax.plot([p[0], p[0] + tick_len * perp[0]], [p[1], p[1] + tick_len * perp[1]], "k-", lw=0.5)
        ax.text(p[0], p[1] - 0.04, f"{int(t*100)}%", ha="center", va="top", fontsize=6)

        # Left edge (el1 -> el2, c=0): top to bottom-left
        p = v_top + t * (v_bl - v_top)
        edge_dir = v_bl - v_top
        perp = np.array([-edge_dir[1], edge_dir[0]])
        perp = perp / np.linalg.norm(perp)
        ax.plot([p[0], p[0] + tick_len * perp[0]], [p[1], p[1] + tick_len * perp[1]], "k-", lw=0.5)
        ax.text(p[0] - 0.04, p[1], f"{int(t*100)}%", ha="right", va="center", fontsize=6)

        # Right edge (el1 -> el3, b=0): top to bottom-right
        p = v_top + t * (v_br - v_top)
        edge_dir = v_br - v_top
        perp = np.array([edge_dir[1], -edge_dir[0]])
        perp = perp / np.linalg.norm(perp)
        ax.plot([p[0], p[0] + tick_len * perp[0]], [p[1], p[1] + tick_len * perp[1]], "k-", lw=0.5)
        ax.text(p[0] + 0.04, p[1], f"{int(t*100)}%", ha="left", va="center", fontsize=6)

    # Triangle outline
    triangle = plt.Polygon([v_top, v_bl, v_br], fill=False, edgecolor="black", lw=1)
    ax.add_patch(triangle)

    # Axis labels (rotated)
    mid_bottom = (v_bl + v_br) / 2
    ax.text(mid_bottom[0], mid_bottom[1] - 0.10, f"{el2} at.%", ha="center", va="top", fontsize=9)

    mid_left = (v_top + v_bl) / 2
    angle_left = math.degrees(math.atan2(*(v_bl - v_top)[::-1]))
    ax.text(mid_left[0] - 0.08, mid_left[1], f"{el1} at.%", ha="center", va="center",
            fontsize=9, rotation=angle_left)

    mid_right = (v_top + v_br) / 2
    angle_right = math.degrees(math.atan2(*(v_br - v_top)[::-1]))
    ax.text(mid_right[0] + 0.08, mid_right[1], f"{el3} at.%", ha="center", va="center",
            fontsize=9, rotation=angle_right)

    # Corner labels
    ax.text(v_top[0], v_top[1] + 0.04, el1, ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(v_bl[0] - 0.04, v_bl[1] - 0.04, el2, ha="right", va="top", fontsize=11, fontweight="bold")
    ax.text(v_br[0] + 0.04, v_br[1] - 0.04, el3, ha="left", va="top", fontsize=11, fontweight="bold")


def plot_system(system: str, ax, contour: bool = False, vmin=None, vmax=None):
    csv_path = OUTPUT_DIR / f"predict_{system}_RMSAD.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No prediction file found: {csv_path}")

    df = pd.read_csv(csv_path)
    elements = _parse_elements(system)
    el1, el2, el3 = elements

    a = df[el1].values
    b = df[el2].values
    c = df[el3].values
    z = df["RMSAD"].values

    x, y = _ternary_to_xy(a, b, c)
    triang = mtri.Triangulation(x, y)

    if contour:
        sc = ax.tricontourf(triang, z, levels=20, cmap="YlOrRd",
                            vmin=vmin, vmax=vmax)
    else:
        sc = ax.tripcolor(triang, z, cmap="YlOrRd", shading="gouraud",
                          vmin=vmin, vmax=vmax)

    _draw_ticks_and_labels(ax, elements)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, math.sqrt(3) / 2 + 0.10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(system, fontsize=12, pad=10)
    return sc


def main():
    parser = argparse.ArgumentParser(description="Plot ternary RMSAD diagrams")
    parser.add_argument("systems", nargs="+", help="System names, e.g. TiNbV TiNbMo")
    parser.add_argument("--contour", action="store_true", help="Use contour fill instead of tripcolor")
    args = parser.parse_args()

    n = len(args.systems)
    fig = plt.figure(figsize=(5 * n + 1, 5))
    gs = gridspec.GridSpec(1, n + 1, width_ratios=[5] * n + [0.3], figure=fig)

    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cax = fig.add_subplot(gs[0, n])

    # Compute shared color range
    all_vals = []
    for system in args.systems:
        csv_path = OUTPUT_DIR / f"predict_{system}_RMSAD.csv"
        if csv_path.exists():
            all_vals.extend(pd.read_csv(csv_path)["RMSAD"].values)
    vmin = min(all_vals) if all_vals else None
    vmax = max(all_vals) if all_vals else None

    sc = None
    for ax, system in zip(axes, args.systems):
        sc = plot_system(system, ax, contour=args.contour, vmin=vmin, vmax=vmax)

    if sc is not None:
        fig.colorbar(sc, cax=cax, label="RMSAD (Å)")

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"ternary_{'_'.join(args.systems)}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
