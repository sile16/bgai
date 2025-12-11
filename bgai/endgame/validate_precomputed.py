

import numpy as np
from pathlib import Path

from bgai.endgame.generator_numba import generate_all_positions
from bgai.endgame.gnubg_reader import (
    compare_with_gnubg_reversed,
    print_comparison_report,
)

OUR_TABLE_PATH = "/home/sile/github/bgai/data/bearoff_15.npy"
GNUBG_DB_PATH = Path(__file__).parent / "gnubg/gnubg_ts10.bd"

def main():
    """Run the validation process."""
    print("Starting validation of precomputed table...")

    # Check if files exist
    if not Path(OUR_TABLE_PATH).exists():
        print(f"ERROR: Our table not found at '{OUR_TABLE_PATH}'")
        return
    if not Path(GNUBG_DB_PATH).exists():
        print(f"ERROR: gnubg database not found at '{GNUBG_DB_PATH}'")
        return

    # 1. Load our precomputed table
    print(f"--- Step 1: Loading our table from {OUR_TABLE_PATH} ---")
    our_table = np.load(OUR_TABLE_PATH)
    our_positions = generate_all_positions(max_checkers=15)

    # 2. Compare our table with the gnubg database
    print(f"\n--- Step 2: Comparing our table with {GNUBG_DB_PATH} ---")
    try:
        report = compare_with_gnubg_reversed(our_table, str(GNUBG_DB_PATH), our_positions)
    except Exception as e:
        print(f"An error occurred during comparison: {e}")
        print("This might be due to an indexing mismatch or file corruption.")
        return

    # 3. Print the results
    print("\n--- Step 3: Comparison Report ---")
    print_comparison_report(report)

if __name__ == "__main__":
    main()

