#!/usr/bin/env python3
"""
Usage: python scripts/run_prediction.py TiNbV TiNbMo ...
"""
import pathlib
import sys

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from rmsad.predict import predict_system  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_prediction.py SYSTEM [SYSTEM ...]")
        print("Example: python scripts/run_prediction.py TiNbV TiNbMo")
        sys.exit(1)

    systems = sys.argv[1:]
    for system in systems:
        print(f"\n=== Predicting RMSAD for {system} ===")
        predict_system(system)


if __name__ == "__main__":
    main()
