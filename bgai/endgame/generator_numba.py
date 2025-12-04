"""Numba-accelerated bearoff table generator with proper move enumeration.

Uses JIT-compiled loops for fast value iteration.
Now with CORRECT minimax over all legal moves (not greedy heuristics).

Memory: 2 full-size arrays (table + new_table) + padded transition array.
"""

import numpy as np
from numba import jit, prange
from typing import Tuple, List, Set
import time
from tqdm import tqdm
import gc

# Defaults for standard bearoff
MAX_CHECKERS_DEFAULT = 15
NUM_POINTS_DEFAULT = 6


# 21 unique dice outcomes (not 36) - we use probabilities to weight
DICE_OUTCOMES = []
DICE_PROBS = []

for d1 in range(1, 7):
    for d2 in range(d1, 7):
        DICE_OUTCOMES.append((d1, d2))
        if d1 == d2:
            DICE_PROBS.append(1/36)  # Doubles
        else:
            DICE_PROBS.append(2/36)  # Non-doubles (2 orderings)

DICE_OUTCOMES = list(DICE_OUTCOMES)
DICE_PROBS = np.array(DICE_PROBS, dtype=np.float32)
N_DICE = len(DICE_OUTCOMES)  # 21

# Maximum moves per dice roll (empirically determined)
# Doubles can have many outcomes due to 4 moves, non-doubles typically fewer
# For 15 checkers with doubles, can exceed 100 outcomes
MAX_MOVES_PER_DICE = 128  # Padded to power of 2 for efficiency


def generate_all_positions(max_checkers: int = MAX_CHECKERS_DEFAULT,
                           num_points: int = NUM_POINTS_DEFAULT) -> List[Tuple[int, ...]]:
    """Generate all bearoff positions as tuples."""
    positions = []

    def gen(remaining: int, points_left: int, current: List[int]):
        if points_left == 1:
            positions.append(tuple(current + [remaining]))
            return
        for v in range(remaining + 1):
            gen(remaining - v, points_left - 1, current + [v])

    for total in range(max_checkers + 1):
        gen(total, num_points, [])

    return positions


def get_legal_moves_for_die(pos: Tuple[int, ...], die: int,
                            num_points: int = NUM_POINTS_DEFAULT) -> List[Tuple[int, ...]]:
    """
    Generates all legal positions resulting from playing a single die.
    Assumes all checkers are in the home board (a bearoff position).
    """
    moves = []
    pos_list = list(pos)

    # Rule 1: Move a checker from a point to a lower point.
    for p in range(die, num_points):
        if pos_list[p] > 0:
            new_pos = pos_list[:]
            new_pos[p] -= 1
            new_pos[p - die] += 1
            moves.append(tuple(new_pos))

    # Rule 2: Bear off a checker from the point corresponding to the die.
    if die - 1 < num_points and pos_list[die - 1] > 0:
        new_pos = pos_list[:]
        new_pos[die - 1] -= 1
        moves.append(tuple(new_pos))

    # Rule 3: If no move under Rule 1 or 2 is possible, you can bear off from a
    # higher point if the die roll is greater than your highest point.
    if not moves:
        highest_occupied = -1
        for i in range(num_points - 1, -1, -1):
            if pos_list[i] > 0:
                highest_occupied = i
                break

        if highest_occupied != -1 and die > highest_occupied + 1:
            new_pos = pos_list[:]
            new_pos[highest_occupied] -= 1
            moves.append(tuple(new_pos))

    return list(set(moves))


def apply_dice(pos: Tuple[int, ...], dice: Tuple[int, int],
               num_points: int = NUM_POINTS_DEFAULT) -> List[Tuple[int, ...]]:
    """
    Generates all legal final positions for a dice roll (d1, d2).
    Handles non-doubles and doubles, and the rule that you must play as
    much of the roll as possible.
    """
    d1, d2 = dice
    if d1 == d2:  # Doubles
        # Apply the die up to 4 times
        positions: Set[Tuple[int, ...]] = {pos}
        for _ in range(4):
            next_positions: Set[Tuple[int, ...]] = set()
            for p in positions:
                moves = get_legal_moves_for_die(p, d1, num_points)
                if moves:
                    next_positions.update(moves)
                else:
                    next_positions.add(p)  # Can't move further
            positions = next_positions
        return list(positions)

    # Non-doubles
    two_move_plays: Set[Tuple[int, ...]] = set()
    one_move_plays: Set[Tuple[int, ...]] = set()

    # Path 1: d1 then d2
    for p1 in get_legal_moves_for_die(pos, d1, num_points):
        moves_d2 = get_legal_moves_for_die(p1, d2, num_points)
        if moves_d2:
            two_move_plays.update(moves_d2)
        else:
            one_move_plays.add(p1)

    # Path 2: d2 then d1
    for p2 in get_legal_moves_for_die(pos, d2, num_points):
        moves_d1 = get_legal_moves_for_die(p2, d1, num_points)
        if moves_d1:
            two_move_plays.update(moves_d1)
        else:
            one_move_plays.add(p2)

    if two_move_plays:
        return list(two_move_plays)

    if one_move_plays:
        return list(one_move_plays)

    return [pos]  # No moves possible


@jit(nopython=True, parallel=True, cache=True)
def value_iteration_step_numba(
    table: np.ndarray,
    new_table: np.ndarray,
    trans: np.ndarray,
    trans_counts: np.ndarray,
    dice_probs: np.ndarray,
    x_done: np.ndarray,
    n: int,
    n_dice: int,
    max_moves: int,
) -> None:
    """Single value iteration step using numba with proper minimax.

    Formula (correct minimax):
    V(X, O) = 1 - E_dice[ min_{X'} V(O, X') ]

    where min is over ALL legal moves X' for that dice roll.
    """
    # Process in parallel over x positions
    for x_idx in prange(n):
        if x_done[x_idx]:
            for o_idx in range(n):
                new_table[x_idx, o_idx] = 1.0
            continue

        for o_idx in range(n):
            if x_done[o_idx]:  # O is done = X loses
                new_table[x_idx, o_idx] = 0.0
                continue

            # V(X, O) = 1 - E_dice[ min_{X'} V(O, X') ]
            expected_min_opp_value = np.float32(0.0)

            for dice_idx in range(n_dice):
                prob = dice_probs[dice_idx]
                n_moves = trans_counts[x_idx, dice_idx]

                # Find minimum opponent value over all legal moves
                min_opp_value = np.float32(1.0)
                for move_idx in range(n_moves):
                    x_new_idx = trans[x_idx, dice_idx, move_idx]

                    if x_done[x_new_idx]:
                        # X borne off -> X wins -> O's value = 0
                        opp_value = np.float32(0.0)
                    else:
                        # V(O, X') from table
                        opp_value = table[o_idx, x_new_idx]

                    if opp_value < min_opp_value:
                        min_opp_value = opp_value

                expected_min_opp_value += prob * min_opp_value

            new_table[x_idx, o_idx] = 1.0 - expected_min_opp_value


def generate_bearoff_table_numba(
    max_checkers: int = MAX_CHECKERS_DEFAULT,
    num_points: int = NUM_POINTS_DEFAULT,
    max_iterations: int = 200,
    tolerance: float = 1e-9,
    output_path: str = None,
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """Generate bearoff table using numba acceleration with proper minimax.

    Uses CORRECT move enumeration - enumerates ALL legal moves and takes
    minimum opponent value (optimal play).

    Memory usage: ~27 GB for 15 checkers (2 x 12 GB tables + transitions)

    Args:
        max_checkers: Maximum checkers per side
        max_iterations: Maximum value iteration steps
        tolerance: Convergence tolerance
        output_path: Path to save final table (optional)

    Returns:
        Tuple of (table, positions)
    """
    print(f"Generating bearoff table for {max_checkers} checkers, {num_points} points (numba + minimax)...")

    positions = generate_all_positions(max_checkers, num_points)
    n = len(positions)
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}

    table_size_gb = n * n * 4 / 1e9
    trans_size_gb = n * N_DICE * MAX_MOVES_PER_DICE * 4 / 1e9
    print(f"  Positions: {n:,}")
    print(f"  Table entries: {n*n:,} ({table_size_gb:.2f} GB)")
    print(f"  Transition array: {trans_size_gb:.2f} GB")
    print(f"  Total memory: ~{2 * table_size_gb + trans_size_gb:.2f} GB")

    # Compute which positions are done (borne off)
    totals = np.array([sum(p) for p in positions], dtype=np.int32)
    x_done = totals == 0

    # Precompute ALL transitions (all legal moves per dice roll)
    print("  Precomputing transitions (all legal moves)...")
    trans = np.zeros((n, N_DICE, MAX_MOVES_PER_DICE), dtype=np.int32)
    trans_counts = np.zeros((n, N_DICE), dtype=np.int32)

    max_moves_seen = 0
    for i, pos in enumerate(tqdm(positions)):
        for j, dice in enumerate(DICE_OUTCOMES):
            new_positions = apply_dice(pos, dice, num_points)
            n_moves = len(new_positions)

            if n_moves > MAX_MOVES_PER_DICE:
                raise ValueError(f"Too many moves ({n_moves}) for position {pos}, dice {dice}")

            max_moves_seen = max(max_moves_seen, n_moves)
            trans_counts[i, j] = n_moves

            for k, new_pos in enumerate(new_positions):
                trans[i, j, k] = pos_to_idx[new_pos]

    print(f"  Max moves per dice roll: {max_moves_seen}")

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
                if x_done[o_idx]:
                    table[x_idx, o_idx] = 0.0

    # Value iteration
    print("  Running value iteration (minimax)...")
    start = time.time()

    for iteration in range(max_iterations):
        iter_start = time.time()

        # Compute new table values (in parallel)
        value_iteration_step_numba(
            table, new_table, trans, trans_counts, DICE_PROBS,
            x_done, n, N_DICE, MAX_MOVES_PER_DICE
        )

        # Compute max_diff outside parallel loop
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
    zero_pos = (0,) * len(positions[0])
    if zero_pos in pos_to_idx:
        idx = pos_to_idx[zero_pos]
        print(f"  V(0, any) should be 1.0: {table[idx, 0]:.6f}")
        if n > 1:
            print(f"  V(any, 0) should be 0.0 (if not 0): {table[1, idx]:.6f}")

    # Simple sanity samples for small tables
    if len(zero_pos) == 6:
        test_positions = [
            ((1, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), "~1.0 (X bears off first)"),
            ((0, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), "first mover advantage"),
            ((0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 1), "~0.81 (need 6 to bear off)"),
        ]

        for x_pos, o_pos, note in test_positions:
            if x_pos in pos_to_idx and o_pos in pos_to_idx:
                val = table[pos_to_idx[x_pos], pos_to_idx[o_pos]]
                print(f"  V({x_pos}, {o_pos}) = {val:.6f} ({note})")

    # Range check
    min_val = table.min()
    max_val = table.max()
    print(f"  Value range: [{min_val:.6f}, {max_val:.6f}]")

    if min_val < -0.001 or max_val > 1.001:
        print("  WARNING: Values outside [0, 1] range!")


if __name__ == "__main__":
    import sys

    max_checkers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    num_points = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_POINTS_DEFAULT

    table, positions = generate_bearoff_table_numba(max_checkers=max_checkers, num_points=num_points)
    verify_table(table, positions)
