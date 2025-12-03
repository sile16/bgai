"""Numba-accelerated bearoff table generator.

Uses JIT-compiled loops for fast value iteration with minimal memory overhead.
Only needs 2 full-size arrays (table + new_table = 24 GB for 15 checkers).
"""

import numpy as np
from numba import jit, prange
from typing import Tuple, List
import time
from tqdm import tqdm
import gc

from .indexing import (
    TOTAL_ONE_SIDED_POSITIONS,
    MAX_CHECKERS,
    NUM_POINTS,
)


# All 36 dice outcomes
N_DICE = 36
DICE_PROB = np.float32(1.0 / 36.0)


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


@jit(nopython=True, parallel=True, cache=True)
def value_iteration_step_numba(
    table: np.ndarray,
    new_table: np.ndarray,
    trans: np.ndarray,
    x_done: np.ndarray,
    o_done: np.ndarray,
    n: int,
    n_dice: int,
) -> None:
    """Single value iteration step using numba.

    Computes new_table from table in-place.
    Note: max_diff must be computed separately after this function returns,
    as prange doesn't support proper reduction across threads.

    Formula:
    V(x, o) = sum_{d1} P(d1) * [
        if trans[x,d1] done: 1.0
        else: sum_{d2} P(d2) * [
            if trans[o,d2] done: 0.0
            else: V(trans[x,d1], trans[o,d2])
        ]
    ]
    """
    prob = np.float32(1.0 / 36.0)

    # Process in parallel over x positions
    for x_idx in prange(n):
        if x_done[x_idx]:
            for o_idx in range(n):
                new_table[x_idx, o_idx] = 1.0
            continue

        x_trans = trans[x_idx]  # Shape: (n_dice,)

        for o_idx in range(n):
            if o_done[o_idx]:
                new_table[x_idx, o_idx] = 0.0
                continue

            o_trans = trans[o_idx]  # Shape: (n_dice,)

            val = np.float32(0.0)
            for d1_idx in range(n_dice):
                x_prime = x_trans[d1_idx]

                if x_done[x_prime]:
                    val += prob * 1.0
                else:
                    inner = np.float32(0.0)
                    for d2_idx in range(n_dice):
                        o_prime = o_trans[d2_idx]
                        if o_done[o_prime]:
                            pass  # inner += prob * 0.0
                        else:
                            inner += prob * table[x_prime, o_prime]
                    val += prob * inner

            new_table[x_idx, o_idx] = val


def generate_bearoff_table_numba(
    max_checkers: int = MAX_CHECKERS,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    output_path: str = None,
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """Generate bearoff table using numba acceleration.

    Memory usage: ~24 GB for 15 checkers (2 x 12 GB arrays).

    Args:
        max_checkers: Maximum checkers per side
        max_iterations: Maximum value iteration steps
        tolerance: Convergence tolerance
        output_path: Path to save final table (optional)

    Returns:
        Tuple of (table, positions)
    """
    print(f"Generating bearoff table for {max_checkers} checkers (numba)...")

    positions = generate_all_positions(max_checkers)
    n = len(positions)
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}

    table_size_gb = n * n * 4 / 1e9
    print(f"  Positions: {n:,}")
    print(f"  Table entries: {n*n:,} ({table_size_gb:.2f} GB)")
    print(f"  Memory usage: ~{2 * table_size_gb:.2f} GB")

    # Compute which positions are done (borne off)
    totals = np.array([sum(p) for p in positions], dtype=np.int32)
    x_done = totals == 0
    o_done = totals == 0

    # Build dice outcomes
    dice_outcomes = []
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            dice_outcomes.append((d1, d2))

    # Precompute transitions
    print("  Precomputing transitions...")
    trans = np.zeros((n, N_DICE), dtype=np.int32)
    for i, pos in enumerate(tqdm(positions)):
        pos_arr = np.array(pos, dtype=np.int32)
        for j, (d1, d2) in enumerate(dice_outcomes):
            new_pos = apply_dice(pos_arr, d1, d2)
            trans[i, j] = pos_to_idx[tuple(new_pos)]

    # Allocate tables
    print("  Allocating tables...")
    table = np.zeros((n, n), dtype=np.float32)
    new_table = np.zeros((n, n), dtype=np.float32)

    # Initialize terminal states
    print("  Initializing terminal states...")
    for x_idx in range(n):
        if x_done[x_idx]:
            table[x_idx, :] = 1.0
        else:
            for o_idx in range(n):
                if o_done[o_idx]:
                    table[x_idx, o_idx] = 0.0

    # Warm up numba JIT
    print("  JIT compiling (first iteration)...")

    # Value iteration
    print("  Running value iteration...")
    start = time.time()

    for iteration in range(max_iterations):
        iter_start = time.time()

        # Compute new table values (in parallel)
        value_iteration_step_numba(
            table, new_table, trans, x_done, o_done, n, N_DICE
        )

        # Compute max_diff outside parallel loop (prange doesn't support reduction)
        max_diff = np.max(np.abs(new_table - table))

        # Swap tables
        table, new_table = new_table, table

        elapsed = time.time() - start
        iter_time = time.time() - iter_start

        if iteration % 5 == 0 or max_diff < tolerance:
            print(f"    Iter {iteration+1}: max_diff={max_diff:.2e}, iter_time={iter_time:.1f}s, total={elapsed:.1f}s")

        if max_diff < tolerance:
            print(f"  Converged after {iteration+1} iterations")
            break

        # Force garbage collection periodically
        if iteration % 10 == 0:
            gc.collect()

    elapsed = time.time() - start
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save if output path provided
    if output_path:
        print(f"  Saving to {output_path}...")
        np.save(output_path, table)

    return table, positions


def verify_table(table: np.ndarray, positions: List[Tuple[int, ...]], n_samples: int = 50):
    """Verify table values are reasonable."""
    n = len(positions)
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}

    print(f"\nVerification ({n_samples} samples):")

    # Test known values
    zero_pos = (0, 0, 0, 0, 0, 0)
    if zero_pos in pos_to_idx:
        idx = pos_to_idx[zero_pos]
        print(f"  V(0, any) should be 1.0: {table[idx, 0]:.6f}")
        if n > 1:
            print(f"  V(any, 0) should be 0.0 (if not 0): {table[1, idx]:.6f}")

    # Symmetric positions should have value ~0.5
    symmetric_tests = [
        ((1, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0)),
        ((0, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0)),
        ((0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 1)),
    ]

    for x_pos, o_pos in symmetric_tests:
        if x_pos in pos_to_idx and o_pos in pos_to_idx:
            val = table[pos_to_idx[x_pos], pos_to_idx[o_pos]]
            print(f"  V({x_pos}, {o_pos}) = {val:.6f} (symmetric, should be ~0.5)")

    # Range check
    min_val = table.min()
    max_val = table.max()
    print(f"  Value range: [{min_val:.6f}, {max_val:.6f}]")

    if min_val < -0.001 or max_val > 1.001:
        print("  WARNING: Values outside [0, 1] range!")


if __name__ == "__main__":
    import sys

    max_checkers = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    table, positions = generate_bearoff_table_numba(max_checkers=max_checkers)
    verify_table(table, positions)

    # Compare with reference for small tables
    if max_checkers <= 5:
        from .generator_v2 import generate_bearoff_table as gen_v2
        print("\nComparing with reference implementation...")
        ref_table, _ = gen_v2(max_checkers=max_checkers, max_iterations=100)
        max_diff = np.max(np.abs(table - ref_table))
        print(f"  Max difference: {max_diff:.2e}")
        print("  PASS!" if max_diff < 1e-5 else "  FAIL!")
