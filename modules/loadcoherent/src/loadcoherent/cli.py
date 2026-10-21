"""Run the fixed reproducible research experiment."""

import argparse

from .study import run_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="examples/results")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    data = run_study(args.output, plots=not args.no_plots)
    print(f"Oracle/base expected MSE: {data['fixed']['oracle']['risk_ratio']:.6f}")
    print(f"Independent KKT discrepancy: {data['kkt_max_error']:.3e} kW")


if __name__ == "__main__":
    main()
