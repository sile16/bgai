"""
Validation script to compare our generated bearoff table with gnubg's.

1. Generates a bearoff table using our reference implementation (generator_v2).
2. Reads a gnubg bearoff database file (.bd).
3. Compares the two tables, accounting for position convention differences.
4. Prints a report of the comparison.
"""

import numpy as np
from pathlib import Path

from .generator_v2 import generate_bearoff_table
from .gnubg_reader import compare_with_gnubg, print_comparison_report

# The gnubg database to compare against.
# From ENDGAME_STATUS.md, this file is for 6 points and 6 checkers.
_SCRIPT_DIR = Path(__file__).parent
GNUBG_DB_PATH = _SCRIPT_DIR / "gnubg/gnubg_ts0.bd"
MAX_CHECKERS_TO_TEST = 6

def main():
    """Run the full validation process."""
    print("Starting validation process...")

    # Check if gnubg database exists
    if not Path(GNUBG_DB_PATH).exists():
        print(f"ERROR: gnubg database not found at '{GNUBG_DB_PATH}'")
        print("Please ensure the file is in the correct location.")
        return

    # 1. Generate our table using the reference implementation
    print(f"\n--- Step 1: Generating our table for {MAX_CHECKERS_TO_TEST} checkers ---")
    our_table, our_positions = generate_bearoff_table(
        max_checkers=MAX_CHECKERS_TO_TEST,
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
