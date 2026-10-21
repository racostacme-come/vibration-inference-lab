"""Command-line entry point for the reproducible study."""

import argparse
import json

from .experiments import run_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reproduced", help="output directory")
    parser.add_argument("--trials", type=int, default=256, help="independent MC trajectories")
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("--trials must be >=2")
    print(json.dumps(run_study(args.output, args.trials), indent=2))


if __name__ == "__main__":
    main()
