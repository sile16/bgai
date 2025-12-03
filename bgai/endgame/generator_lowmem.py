"""Low-memory bearoff table generator using memory-mapped arrays.

Uses numpy.memmap for disk-backed arrays to handle tables larger than RAM.
Only keeps one working row in memory at a time.
"""

import numpy as np
from typing import Tuple, List
import time
from tqdm import tqdm
import os
import gc

from .indexing import (
    TOTAL_ONE_SIDED_POSITIONS,
    MAX_CHECKERS,
    NUM_POINTS,
)


# All 36 dice outcomes
ALL_DICE = [(d1, d2) for d1 in range(1, 7) for d2 in range(1, 7)]
DICE_PROB = 1.0 / 36.0
N_DICE = len(ALL_DICE)


def generate_all_positions(max_checkers: int = MAX_CHECKERS) -> List[Tuple[int, ...]]:
    """Generate all bearoff positions as tuples."""
    positions = []

    def gen(remaining: int, points_left: int, current: List[int]):
        if points_left == 1:
            positions.append(tuple(current + [remaining]))
            return
        for v in range(remaining + 1):
            gen(remaining - v, points_left - 1, current + [v])

    for total in range(max_checkers + 1):
        gen(total, NUM_POINTS, [])

    return positions


def apply_single_die(pos: np.ndarray, die: int) -> np.ndarray:
    """Apply single die move optimally in bearoff."""
    pos = pos.copy()
    total = pos.sum()

    if total == 0:
        return pos

    if pos[die - 1] > 0:
        pos[die - 1] -= 1
        return pos

    highest = -1
    for i in range(5, -1, -1):
        if pos[i] > 0:
            highest = i
            break

    if highest < 0:
        return pos

    if die > highest + 1:
        pos[highest] -= 1
        return pos

    for src in range(5, -1, -1):
        dst = src - die
        if pos[src] > 0 and dst >= 0:
            pos[src] -= 1
            pos[dst] += 1
            return pos

    return pos


def apply_dice(pos: np.ndarray, d1: int, d2: int) -> np.ndarray:
    """Apply a dice roll (d1, d2) optimally."""
    if d1 == d2:
        for _ in range(4):
            pos = apply_single_die(pos, d1)
    else:
        pos1 = apply_single_die(pos.copy(), d1)
        pos1 = apply_single_die(pos1, d2)

        pos2 = apply_single_die(pos.copy(), d2)
        pos2 = apply_single_die(pos2, d1)

        pos = pos1 if pos1.sum() <= pos2.sum() else pos2

    return pos


def generate_bearoff_table_lowmem(
    max_checkers: int = MAX_CHECKERS,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    output_path: str = None,
    temp_dir: str = "/tmp",
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """Generate bearoff table using memory-mapped files.

    Uses disk-backed storage to handle tables larger than RAM.
    Peak memory usage: ~2 * n * 4 bytes (two rows) + transitions array.

    Args:
        max_checkers: Maximum checkers per side
        max_iterations: Maximum value iteration steps
        tolerance: Convergence tolerance
        output_path: Path to save final table (optional)
        temp_dir: Directory for temporary files

    Returns:
        Tuple of (table, positions)
    """
    print(f"Generating bearoff table for {max_checkers} checkers (low-memory mode)...")

    positions = generate_all_positions(max_checkers)
    n = len(positions)
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}

    table_size_gb = n * n * 4 / 1e9
    print(f"  Positions: {n:,}")
    print(f"  Table entries: {n*n:,} ({table_size_gb:.2f} GB)")

    # Compute which positions are done (borne off)
    totals = np.array([sum(p) for p in positions], dtype=np.int32)
    x_done = totals == 0

    # Precompute transitions
    print("  Precomputing transitions...")
    trans = np.zeros((n, N_DICE), dtype=np.int32)
    for i, pos in enumerate(tqdm(positions)):
        pos_arr = np.array(pos, dtype=np.int32)
        for j, (d1, d2) in enumerate(ALL_DICE):
            new_pos = apply_dice(pos_arr, d1, d2)
            trans[i, j] = pos_to_idx[tuple(new_pos)]

    # Create memory-mapped files for tables
    table_file = os.path.join(temp_dir, f"bearoff_table_{max_checkers}.dat")
    new_table_file = os.path.join(temp_dir, f"bearoff_new_table_{max_checkers}.dat")

    print(f"  Creating memory-mapped files in {temp_dir}...")

    # Initialize tables
    table = np.memmap(table_file, dtype=np.float32, mode='w+', shape=(n, n))
    new_table = np.memmap(new_table_file, dtype=np.float32, mode='w+', shape=(n, n))

    # Initialize terminal states
    print("  Initializing terminal states...")
    for x_idx in range(n):
        if x_done[x_idx]:
            table[x_idx, :] = 1.0
        else:
            for o_idx in range(n):
                if x_done[o_idx]:  # O is done
                    table[x_idx, o_idx] = 0.0
    table.flush()

    # Value iteration
    print("  Running value iteration...")
    start = time.time()

    for iteration in range(max_iterations):
        max_diff = 0.0

        # Process row by row to minimize memory
        for x_idx in tqdm(range(n), desc=f"Iter {iteration+1}", disable=iteration > 0):
            if x_done[x_idx]:
                new_table[x_idx, :] = 1.0
                continue

            # Get transitions for this x
            x_trans = trans[x_idx]  # Shape: (N_DICE,)

            # Compute new values for this row
            new_row = np.zeros(n, dtype=np.float32)

            for o_idx in range(n):
                if x_done[o_idx]:  # O is done
                    new_row[o_idx] = 0.0
                    continue

                # V(x, o) = sum_{d1} P(d1) * [
                #   if x' done: 1.0
                #   else: sum_{d2} P(d2) * [
                #     if o' done: 0.0
                #     else: V(x', o')
                #   ]
                # ]
                o_trans = trans[o_idx]  # Shape: (N_DICE,)

                val = 0.0
                for d1_idx in range(N_DICE):
                    x_prime = x_trans[d1_idx]

                    if x_done[x_prime]:
                        val += DICE_PROB * 1.0
                    else:
                        inner = 0.0
                        for d2_idx in range(N_DICE):
                            o_prime = o_trans[d2_idx]
                            if x_done[o_prime]:  # O borne off
                                inner += DICE_PROB * 0.0
                            else:
                                inner += DICE_PROB * table[x_prime, o_prime]
                        val += DICE_PROB * inner

                new_row[o_idx] = val

            # Compute diff before writing
            old_row = table[x_idx, :].copy()
            row_diff = np.max(np.abs(new_row - old_row))
            max_diff = max(max_diff, row_diff)

            new_table[x_idx, :] = new_row

        new_table.flush()

        # Swap tables (swap file references)
        table, new_table = new_table, table
        table_file, new_table_file = new_table_file, table_file

        elapsed = time.time() - start
        if iteration % 5 == 0 or max_diff < tolerance:
            print(f"    Iter {iteration+1}: max_diff={max_diff:.2e}, elapsed={elapsed:.1f}s")

        if max_diff < tolerance:
            print(f"  Converged after {iteration+1} iterations")
            break

    elapsed = time.time() - start
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Copy to regular numpy array for return
    print("  Copying result to memory...")
    result = np.array(table, dtype=np.float32)

    # Clean up temp files
    del table
    del new_table
    gc.collect()

    try:
        os.remove(table_file)
        os.remove(new_table_file)
    except:
        pass

    # Save if output path provided
    if output_path:
        print(f"  Saving to {output_path}...")
        np.save(output_path, result)

    return result, positions


def verify_table(table: np.ndarray, positions: List[Tuple[int, ...]], n_samples: int = 50):
    """Verify table values are reasonable."""
    n = len(positions)
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}
    totals = np.array([sum(p) for p in positions])

    print(f"\nVerification ({n_samples} samples):")

    # Test known values
    zero_pos = (0, 0, 0, 0, 0, 0)
    if zero_pos in pos_to_idx:
        idx = pos_to_idx[zero_pos]
        print(f"  V(0, any) should be 1.0: {table[idx, 0]:.6f}")
        print(f"  V(any, 0) should be 0.0 (if not 0): {table[1, idx]:.6f}")

    # Symmetric positions
    symmetric_tests = [
        ((1, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0)),
        ((0, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0)),
        ((0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 1)),
    ]

    for x_pos, o_pos in symmetric_tests:
        if x_pos in pos_to_idx and o_pos in pos_to_idx:
            val = table[pos_to_idx[x_pos], pos_to_idx[o_pos]]
            print(f"  V({x_pos}, {o_pos}) = {val:.6f} (should be ~0.5 by symmetry)")

    # Range check
    min_val = table.min()
    max_val = table.max()
    print(f"  Value range: [{min_val:.6f}, {max_val:.6f}]")

    if min_val < -0.001 or max_val > 1.001:
        print("  WARNING: Values outside [0, 1] range!")


if __name__ == "__main__":
    # Test with small table
    table, positions = generate_bearoff_table_lowmem(max_checkers=5)
    verify_table(table, positions)
