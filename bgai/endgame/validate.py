"""
Validation script to compare our generated bearoff table with gnubg's.

1. Generates a bearoff table using the fast numba implementation.
2. Reads a gnubg bearoff database file (.bd).
3. Compares the two tables, accounting for position convention differences.
4. Prints a report of the comparison.
"""

import numpy as np
from pathlib import Path

from .generator_numba import generate_bearoff_table_numba
from .gnubg_reader import (
    compare_with_gnubg,
    print_comparison_report,
    read_gnubg_header,
)

# The gnubg database to compare against.
# This file ships with 5 points and 7 checkers (header: gnubg-TS-05-07-1...).
_SCRIPT_DIR = Path(__file__).parent
GNUBG_DB_PATH = _SCRIPT_DIR / "gnubg/board_t57.bd"

def main():
    """Run the full validation process."""
    print("Starting validation process...")

    # Check if gnubg database exists
    if not Path(GNUBG_DB_PATH).exists():
        print(f"ERROR: gnubg database not found at '{GNUBG_DB_PATH}'")
        print("Please ensure the file is in the correct location.")
        return

    # Read metadata from the gnubg file so we match points/checker count
    n_points, max_checkers, header_str = read_gnubg_header(str(GNUBG_DB_PATH))
    print(f"Found gnubg database: {header_str}")
    print(f"  Points: {n_points}, Max checkers: {max_checkers}")

    # 1. Generate our table with numba
    print(f"\n--- Step 1: Generating our table for {max_checkers} checkers, {n_points} points ---")
    our_table, our_positions = generate_bearoff_table_numba(
        max_checkers=max_checkers,
        num_points=n_points,
        max_iterations=100,
        tolerance=1e-8
    )

    # 2. Compare our table with the gnubg database
    print("\n--- Step 2: Comparing our table with gnubg ---")
    try:
        report = compare_with_gnubg(our_table, GNUBG_DB_PATH, our_positions)
    except Exception as e:
        print(f"An error occurred during comparison: {e}")
        print("This might be due to an indexing mismatch or file corruption.")
        return

    # 3. Print the results
    print("\n--- Step 3: Comparison Report ---")
    print_comparison_report(report)

if __name__ == "__main__":
    main()
