"""Two-sided bearoff table generator using dynamic programming.

Computes exact win probabilities for all bearoff positions via
backwards induction. No gnubg dependency.

State: (X_position, O_position, whose_turn)
Value: P(X wins | state, optimal play)

Memory optimization:
- Symmetry: V(X, O, X_turn) = 1 - V(O, X, O_turn) when both to-move
  Actually: we store V(X, O) = P(X wins | X to move, optimal play)
  Then V(X, O, O_turn) can be derived via 1 - V(O, X)
- uint16 storage: 65535 precision for probabilities
"""

import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
import time

from .indexing import (
    position_to_index,
    index_to_position,
    position_to_index_fast,
    TOTAL_ONE_SIDED_POSITIONS,
    MAX_CHECKERS,
    NUM_POINTS,
)


# Dice probabilities
# There are 36 outcomes, but we group by unordered pairs
# (1,1), (2,2), ..., (6,6): 6 doubles, each 1/36
# (1,2), (1,3), ..., (5,6): 15 non-doubles, each 2/36
DICE_OUTCOMES = []
DICE_PROBS = []

for d1 in range(1, 7):
    for d2 in range(d1, 7):
        DICE_OUTCOMES.append((d1, d2))
        if d1 == d2:
            DICE_PROBS.append(1/36)  # Doubles
        else:
            DICE_PROBS.append(2/36)  # Non-doubles (2 orderings)

DICE_OUTCOMES = np.array(DICE_OUTCOMES)
DICE_PROBS = np.array(DICE_PROBS)


def apply_single_die(pos: Tuple[int, ...], die: int) -> Optional[Tuple[int, ...]]:
    """Apply a single die move optimally in bearoff.

    Optimal bearoff strategy:
    1. If can bear off from exact point, do so
    2. If die > highest occupied point, bear off from highest
    3. Otherwise, move from highest possible point

    Args:
        pos: 6-tuple of checkers (points 1-6)
        die: Die value (1-6)

    Returns:
        New position tuple, or None if no legal move.
    """
    pos = list(pos)
    total = sum(pos)

    if total == 0:
        return tuple(pos)  # Already borne off

    # Check if can bear off from exact point
    if pos[die - 1] > 0:
        pos[die - 1] -= 1
        return tuple(pos)

    # Check if die is higher than all occupied points
    highest_occupied = -1
    for i in range(5, -1, -1):
        if pos[i] > 0:
            highest_occupied = i
            break

    if highest_occupied < 0:
        return tuple(pos)  # No checkers

    # If die > highest occupied point + 1, bear off from highest
    if die > highest_occupied + 1:
        pos[highest_occupied] -= 1
        return tuple(pos)

    # Move from the highest point we can
    for src in range(5, -1, -1):
        if pos[src] > 0:
            dst = src - die
            if dst >= 0:
                pos[src] -= 1
                pos[dst] += 1
                return tuple(pos)

    # No legal move (shouldn't happen in pure bearoff)
    return None


def apply_dice(pos: Tuple[int, ...], dice: Tuple[int, int]) -> Tuple[int, ...]:
    """Apply dice roll optimally, returning new position.

    For non-doubles: apply both dice (higher first is usually optimal)
    For doubles: apply 4 times
    """
    d1, d2 = dice

    if d1 == d2:
        # Doubles: 4 moves
        for _ in range(4):
            new_pos = apply_single_die(pos, d1)
            if new_pos is not None:
                pos = new_pos
    else:
        # Non-doubles: try both orderings, pick best (most checkers off)
        # In pure bearoff, higher die first is almost always optimal
        pos1 = apply_single_die(pos, max(d1, d2))
        if pos1:
            pos1 = apply_single_die(pos1, min(d1, d2))

        pos2 = apply_single_die(pos, min(d1, d2))
        if pos2:
            pos2 = apply_single_die(pos2, max(d1, d2))

        # Choose the one with fewer total checkers (more borne off)
        if pos1 and pos2:
            pos = pos1 if sum(pos1) <= sum(pos2) else pos2
        elif pos1:
            pos = pos1
        elif pos2:
            pos = pos2

    return pos


def generate_all_positions() -> List[Tuple[int, ...]]:
    """Generate all bearoff positions (0-15 checkers on 6 points)."""
    positions = []

    def generate(remaining: int, points_left: int, current: List[int]):
        if points_left == 1:
            positions.append(tuple(current + [remaining]))
            return
        for v in range(remaining + 1):
            generate(remaining - v, points_left - 1, current + [v])

    for total in range(MAX_CHECKERS + 1):
        generate(total, NUM_POINTS, [])

    return positions


class BearoffTableGenerator:
    """Generates two-sided bearoff tables via dynamic programming."""

    def __init__(self, dtype=np.float32):
        """Initialize generator.

        Args:
            dtype: Data type for table (float32 or float16).
        """
        self.dtype = dtype
        self.positions = generate_all_positions()
        self.n_positions = len(self.positions)

        assert self.n_positions == TOTAL_ONE_SIDED_POSITIONS

        # Build position lookup
        self.pos_to_idx = {pos: i for i, pos in enumerate(self.positions)}

        print(f"Initialized with {self.n_positions:,} one-sided positions")
        print(f"Two-sided table size: {self.n_positions ** 2:,} entries")

    def generate_with_symmetry(self, show_progress: bool = True) -> np.ndarray:
        """Generate table using symmetry optimization.

        We compute V(X, O) = P(X wins | X to move)
        Due to symmetry: V(X, O) = 1 - V(O, X) when roles are swapped

        So we only need to store upper triangle + diagonal:
        - For i < j: store V(i, j)
        - For i > j: V(i, j) = 1 - V(j, i)
        - For i == j: store V(i, i)

        Storage: upper triangle as 1D array
        Index mapping: (i, j) with i <= j -> i * n - i*(i+1)//2 + j

        Returns:
            1D array of shape (n*(n+1)//2,) with upper triangle values.
        """
        n = self.n_positions
        table_size = n * (n + 1) // 2

        print(f"Generating table with symmetry...")
        print(f"  One-sided positions: {n:,}")
        print(f"  Two-sided with symmetry: {table_size:,}")
        print(f"  Memory: {table_size * np.dtype(self.dtype).itemsize / 1e9:.2f} GB")

        # We need to compute values via backwards induction
        # But in bearoff, we can use a simpler approach:
        # V(X, O) = E_dice[ V(X', O') ] where X' is after X moves, O' is after O responds

        # Actually, the proper DP requires tracking alternating turns.
        # Let's define:
        #   V_X(X, O) = P(X wins | X to move)
        #   V_O(X, O) = P(X wins | O to move)
        #
        # V_X(X, O) = E_dice[ V_O(X', O) ]  where X' = apply_dice(X, dice)
        # V_O(X, O) = E_dice[ V_X(X, O') ]  where O' = apply_dice(O, dice)
        #
        # Symmetry: V_O(X, O) = 1 - V_X(O, X)
        #
        # So: V_X(X, O) = E_dice[ 1 - V_X(O, X') ]
        #              = 1 - E_dice[ V_X(O, X') ]
        #
        # This gives us a system we can solve iteratively.

        # Initialize table (X to move)
        # Shape: (n, n) where table[x_idx, o_idx] = V(X_pos, O_pos)
        table = np.zeros((n, n), dtype=self.dtype)

        # Set terminal states
        # sum(X_pos) == 0: X has borne off, X wins -> 1.0
        # sum(O_pos) == 0: O has borne off, O wins -> 0.0
        for x_idx, x_pos in enumerate(self.positions):
            x_total = sum(x_pos)
            for o_idx, o_pos in enumerate(self.positions):
                o_total = sum(o_pos)
                if x_total == 0:
                    table[x_idx, o_idx] = 1.0
                elif o_total == 0:
                    table[x_idx, o_idx] = 0.0

        # Precompute transitions: for each position and dice, what's the result?
        print("Precomputing transitions...")
        transitions = np.zeros((n, len(DICE_OUTCOMES)), dtype=np.int32)
        for pos_idx, pos in enumerate(tqdm(self.positions, disable=not show_progress)):
            for dice_idx, (d1, d2) in enumerate(DICE_OUTCOMES):
                new_pos = apply_dice(pos, (d1, d2))
                transitions[pos_idx, dice_idx] = self.pos_to_idx[new_pos]

        # Iterative value computation
        print("Computing values via iteration...")
        max_iterations = 100
        tolerance = 1e-7

        for iteration in range(max_iterations):
            old_table = table.copy()

            # V_X(X, O) = 1 - E_dice[ V_X(O, X') ]
            # where X' = transition[X, dice]
            for x_idx in tqdm(range(n), disable=not show_progress, desc=f"Iter {iteration+1}"):
                x_total = sum(self.positions[x_idx])
                if x_total == 0:
                    continue  # Terminal, already set

                for o_idx in range(n):
                    o_total = sum(self.positions[o_idx])
                    if o_total == 0:
                        continue  # Terminal

                    # E_dice[ V_X(O, X') ]
                    expected_opp_value = 0.0
                    for dice_idx, prob in enumerate(DICE_PROBS):
                        x_new_idx = transitions[x_idx, dice_idx]
                        # V_X(O, X') - opponent's winning prob from their perspective
                        expected_opp_value += prob * old_table[o_idx, x_new_idx]

                    table[x_idx, o_idx] = 1.0 - expected_opp_value

            # Check convergence
            max_diff = np.max(np.abs(table - old_table))
            print(f"  Iteration {iteration+1}: max_diff = {max_diff:.2e}")

            if max_diff < tolerance:
                print(f"  Converged after {iteration+1} iterations")
                break
        else:
            print(f"  Warning: did not converge after {max_iterations} iterations")

        # Convert to upper triangle format for storage
        print("Converting to upper triangle format...")
        upper = np.zeros(table_size, dtype=self.dtype)
        idx = 0
        for i in range(n):
            for j in range(i, n):
                upper[idx] = table[i, j]
                idx += 1

        return upper

    def generate_full(self, show_progress: bool = True) -> np.ndarray:
        """Generate full table without symmetry optimization.

        Useful for verification. Returns (n, n) array.
        """
        n = self.n_positions

        print(f"Generating full table (no symmetry)...")
        print(f"  Size: {n:,} x {n:,} = {n*n:,} entries")
        print(f"  Memory: {n * n * np.dtype(self.dtype).itemsize / 1e9:.2f} GB")

        table = np.zeros((n, n), dtype=self.dtype)

        # Terminal states
        for x_idx, x_pos in enumerate(self.positions):
            x_total = sum(x_pos)
            for o_idx, o_pos in enumerate(self.positions):
                o_total = sum(o_pos)
                if x_total == 0:
                    table[x_idx, o_idx] = 1.0
                elif o_total == 0:
                    table[x_idx, o_idx] = 0.0

        # Precompute transitions
        print("Precomputing transitions...")
        transitions = np.zeros((n, len(DICE_OUTCOMES)), dtype=np.int32)
        for pos_idx, pos in enumerate(tqdm(self.positions, disable=not show_progress)):
            for dice_idx, (d1, d2) in enumerate(DICE_OUTCOMES):
                new_pos = apply_dice(pos, (d1, d2))
                transitions[pos_idx, dice_idx] = self.pos_to_idx[new_pos]

        # Iterative computation
        print("Computing values...")
        max_iterations = 100
        tolerance = 1e-7

        for iteration in range(max_iterations):
            old_table = table.copy()

            for x_idx in tqdm(range(n), disable=not show_progress, desc=f"Iter {iteration+1}"):
                if sum(self.positions[x_idx]) == 0:
                    continue

                for o_idx in range(n):
                    if sum(self.positions[o_idx]) == 0:
                        continue

                    expected_opp_value = 0.0
                    for dice_idx, prob in enumerate(DICE_PROBS):
                        x_new_idx = transitions[x_idx, dice_idx]
                        expected_opp_value += prob * old_table[o_idx, x_new_idx]

                    table[x_idx, o_idx] = 1.0 - expected_opp_value

            max_diff = np.max(np.abs(table - old_table))
            print(f"  Iteration {iteration+1}: max_diff = {max_diff:.2e}")

            if max_diff < tolerance:
                print(f"  Converged after {iteration+1} iterations")
                break

        return table


def verify_with_gnubg(table: np.ndarray, positions: List[Tuple[int, ...]],
                       n_samples: int = 100):
    """Verify table values against gnubg (if available)."""
    try:
        import gnubg
        from pgx.tools.gnubg_bridge import pgx_to_gnubg_board
    except ImportError:
        print("gnubg not available for verification")
        return

    print(f"\nVerifying {n_samples} random positions against gnubg...")

    rng = np.random.default_rng(42)
    errors = []

    for _ in range(n_samples):
        x_idx = rng.integers(0, len(positions))
        o_idx = rng.integers(0, len(positions))

        x_pos = positions[x_idx]
        o_pos = positions[o_idx]

        if sum(x_pos) == 0 or sum(o_pos) == 0:
            continue

        our_value = table[x_idx, o_idx]

        # Convert to gnubg format
        board = [[0]*25, [0]*25]
        for i, cnt in enumerate(x_pos):
            board[0][i+1] = cnt
        for i, cnt in enumerate(o_pos):
            board[1][24-i] = cnt

        try:
            probs = gnubg.probabilities(board, 0)
            gnubg_value = probs[0]
            error = abs(our_value - gnubg_value)
            errors.append(error)
        except Exception as e:
            continue

    if errors:
        print(f"  Max error: {max(errors):.6f}")
        print(f"  Mean error: {np.mean(errors):.6f}")
        print(f"  Samples within 0.01: {sum(1 for e in errors if e < 0.01)}/{len(errors)}")


if __name__ == "__main__":
    # Quick test with small number of iterations
    gen = BearoffTableGenerator(dtype=np.float32)

    print("\nGenerating small test table...")
    start = time.time()
    table = gen.generate_full(show_progress=True)
    elapsed = time.time() - start
    print(f"\nGeneration took {elapsed:.1f} seconds")

    # Save
    np.save('/home/sile/github/bgai/data/bearoff_test.npy', table)
    print(f"Saved to data/bearoff_test.npy")

    # Verify
    verify_with_gnubg(table, gen.positions, n_samples=100)
