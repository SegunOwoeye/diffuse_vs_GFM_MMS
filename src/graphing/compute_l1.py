"""Compatibility CLI for legacy graphing-side L1/error tables."""

import sys

import numpy as np

from error_metrics import compute_errors, save_error_table


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python compute_l1.py folder_name")
        raise SystemExit(1)

    folder = sys.argv[1]
    results = compute_errors(folder)

    print("\nL1 Errors and Observed L1 Orders:")
    for row in results:
        order = row["order_rho_L1"]
        order_text = "nan" if np.isnan(order) else f"{order:.3f}"
        print(
            f"N={row['N']} | "
            f"rho={row['rho_L1']:.4e} "
            f"u={row['u_L1']:.4e} "
            f"p={row['p_L1']:.4e} "
            f"e={row['e_L1']:.4e} "
            f"| O_rho={order_text}"
        )

    save_error_table(results, folder)


if __name__ == "__main__":
    main()
