#!/usr/bin/env python3
"""
Usage:
  python scripts/run_prediction.py TiNbV TiNbMo
  python scripts/run_prediction.py TiNbVMo --step 5
  python scripts/run_prediction.py TiNbV --dparameter-dir ~/Research/dparameter/data/output
"""
import argparse
import pathlib
import sys

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from rmsad.predict import predict_system  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Predict RMSAD, shear modulus, and (optionally) yield strength over a full composition grid."
    )
    parser.add_argument("systems", nargs="+", help="System names, e.g. TiNbV TiNbVMo")
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        metavar="PCT",
        help="Composition step size in at.%% (default: 1). "
             "Recommended: 1 for binary/ternary, 5 for quaternary, 10 for quinary+.",
    )
    parser.add_argument(
        "--dparameter-dir",
        type=pathlib.Path,
        default=None,
        metavar="DIR",
        help="Path to d-parameter output directory containing predict_{system}_HTP.csv files. "
             "If provided, gamma_usf is merged and YS_GPa is computed.",
    )
    args = parser.parse_args()

    if args.step < 1 or 100 % args.step != 0:
        parser.error(f"--step must be a divisor of 100 (got {args.step})")

    for system in args.systems:
        print(f"\n=== Predicting for {system} (step={args.step}%) ===")
        predict_system(system, step=args.step, dparameter_dir=args.dparameter_dir)


if __name__ == "__main__":
    main()
