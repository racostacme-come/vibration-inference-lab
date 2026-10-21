"""Reproduce AliasGuard's analytical vibration experiment."""

import argparse
import json

from .study import run_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="examples/results", help="directory for JSON/CSV/figures"
    )
    parser.add_argument("--no-plots", action="store_true", help="write numerical results only")
    args = parser.parse_args()
    result = run_study(args.output, plots=not args.no_plots)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
