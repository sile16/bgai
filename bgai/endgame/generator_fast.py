"""Fast JAX-accelerated bearoff table generator.

Uses vectorized operations and JIT compilation for speed.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from typing import Tuple, List
import time
from tqdm import tqdm

from .indexing import (
    TOTAL_ONE_SIDED_POSITIONS,
    MAX_CHECKERS,
    NUM_POINTS,
    binomial,
    positions_for_checkers,
)


def generate_all_positions_array(max_checkers: int = MAX_CHECKERS) -> np.ndarray:
    """Generate all positions as numpy array.

    Returns:
        Array of shape (n_positions, 6) with checker counts.
    """
    positions = []

    def generate(remaining: int, points_left: int, current: List[int]):
        if points_left == 1:
            positions.append(current + [remaining])
            return
        for v in range(remaining + 1):
            generate(remaining - v, points_left - 1, current + [v])

    for total in range(max_checkers + 1):
        generate(total, NUM_POINTS, [])

    return np.array(positions, dtype=np.int32)


# Precompute dice outcomes and probabilities
DICE_OUTCOMES_NP = np.array([
    (d1, d2) for d1 in range(1, 7) for d2 in range(d1, 7)
], dtype=np.int32)

DICE_PROBS_NP = np.array([
    1/36 if d1 == d2 else 2/36
    for d1 in range(1, 7) for d2 in range(d1, 7)
], dtype=np.float32)


@jax.jit
def apply_single_die_jax(pos: jnp.ndarray, die: int) -> jnp.ndarray:
    """Apply single die move optimally (JAX version).

    Args:
        pos: 6-element array of checker counts
        die: Die value (1-6)

    Returns:
        New position array
    """
    total = jnp.sum(pos)

    # Find highest occupied point
    occupied_mask = pos > 0
    highest = jnp.where(occupied_mask, jnp.arange(6), -1).max()

    # Case 1: Can bear off from exact point
    can_bearoff_exact = pos[die - 1] > 0
    pos_bearoff_exact = pos.at[die - 1].add(-1)

    # Case 2: Die > highest + 1, bear off from highest
    can_bearoff_high = (die > highest + 1) & (highest >= 0)
    pos_bearoff_high = jnp.where(
        highest >= 0,
        pos.at[highest].add(-1),
        pos
    )

    # Case 3: Move from highest point that can move
    def find_moveable(pos, die):
        # Find highest point that can legally move
        for src in range(5, -1, -1):
            dst = src - die
            if pos[src] > 0 and dst >= 0:
                return src, dst
        return -1, -1

    # We can't use Python loops in JIT, so we handle this differently
    # Check each source point
    src_options = jnp.arange(5, -1, -1)  # [5,4,3,2,1,0]
    dst_options = src_options - die

    valid_moves = (pos[src_options] > 0) & (dst_options >= 0)
    # Get first valid move (highest src)
    first_valid_idx = jnp.argmax(valid_moves)
    has_valid_move = valid_moves.any()

    src = jnp.where(has_valid_move, src_options[first_valid_idx], -1)
    dst = jnp.where(has_valid_move, dst_options[first_valid_idx], -1)

    pos_move = jnp.where(
        has_valid_move,
        pos.at[src].add(-1).at[dst].add(1),
        pos
    )

    # Select appropriate result
    result = jnp.where(total == 0, pos,
             jnp.where(can_bearoff_exact, pos_bearoff_exact,
             jnp.where(can_bearoff_high, pos_bearoff_high,
             jnp.where(has_valid_move, pos_move, pos))))

    return result


def apply_dice_numpy(pos: np.ndarray, d1: int, d2: int) -> np.ndarray:
    """Apply dice roll optimally (numpy version for precomputation).

    Faster than JAX for single positions during precomputation.
    """
    pos = pos.copy()

    def apply_one(pos, die):
        total = pos.sum()
        if total == 0:
            return pos

        # Bear off from exact point?
        if pos[die - 1] > 0:
            pos[die - 1] -= 1
            return pos

        # Find highest occupied
        highest = -1
        for i in range(5, -1, -1):
            if pos[i] > 0:
                highest = i
                break

        if highest < 0:
            return pos

        # Bear off from highest if die is too big?
        if die > highest + 1:
            pos[highest] -= 1
            return pos

        # Move from highest possible
        for src in range(5, -1, -1):
            dst = src - die
            if pos[src] > 0 and dst >= 0:
                pos[src] -= 1
                pos[dst] += 1
                return pos

        return pos

    if d1 == d2:
        # Doubles: 4 moves
        for _ in range(4):
            pos = apply_one(pos, d1)
    else:
        # Try higher die first
        pos1 = apply_one(pos.copy(), max(d1, d2))
        pos1 = apply_one(pos1, min(d1, d2))

        # Try lower die first
        pos2 = apply_one(pos.copy(), min(d1, d2))
        pos2 = apply_one(pos2, max(d1, d2))

        # Choose better result (fewer checkers)
        pos = pos1 if pos1.sum() <= pos2.sum() else pos2

    return pos


def precompute_transitions(positions: np.ndarray) -> np.ndarray:
    """Precompute transition table.

    Args:
        positions: Array of shape (n_positions, 6)

    Returns:
        Array of shape (n_positions, 21) with new position indices
    """
    n = positions.shape[0]
    n_dice = DICE_OUTCOMES_NP.shape[0]

    # Build position to index lookup
    pos_to_idx = {}
    for i, pos in enumerate(positions):
        pos_to_idx[tuple(pos)] = i

    transitions = np.zeros((n, n_dice), dtype=np.int32)

    for i, pos in enumerate(tqdm(positions, desc="Precomputing transitions")):
        for j, (d1, d2) in enumerate(DICE_OUTCOMES_NP):
            new_pos = apply_dice_numpy(pos, int(d1), int(d2))
            transitions[i, j] = pos_to_idx[tuple(new_pos)]

    return transitions


@partial(jax.jit, static_argnums=(2,))
def value_iteration_step(table: jnp.ndarray, transitions: jnp.ndarray,
                          n_dice: int) -> jnp.ndarray:
    """Single step of value iteration (JAX-accelerated).

    V(x, o) = 1 - E_dice[ V(o, x') ]
    where x' = transitions[x, dice]

    Args:
        table: Current value table (n, n)
        transitions: Transition indices (n, n_dice)
        n_dice: Number of dice outcomes (static)

    Returns:
        Updated value table
    """
    n = table.shape[0]

    # For each (x, o) pair, compute the new value
    # new_V[x, o] = 1 - sum_dice P(dice) * V[o, transitions[x, dice]]

    def compute_value(x_idx, o_idx):
        # Get all next x positions for this x
        x_next_indices = transitions[x_idx]  # Shape: (n_dice,)

        # Look up V(o, x') for each dice outcome
        opp_values = table[o_idx, x_next_indices]  # Shape: (n_dice,)

        # Expected opponent value
        expected_opp = jnp.dot(DICE_PROBS_NP, opp_values)

        return 1.0 - expected_opp

    # Vectorize over all positions
    # This creates a (n, n) grid of values
    x_grid, o_grid = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing='ij')

    new_table = jax.vmap(jax.vmap(compute_value))(x_grid, o_grid)

    return new_table


def generate_bearoff_table_jax(max_checkers: int = MAX_CHECKERS,
                                 max_iterations: int = 100,
                                 tolerance: float = 1e-7,
                                 dtype: np.dtype = np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """Generate bearoff table using JAX acceleration.

    Args:
        max_checkers: Maximum checkers per side
        max_iterations: Maximum value iteration steps
        tolerance: Convergence tolerance
        dtype: Output data type

    Returns:
        Tuple of (table, positions) where:
        - table: (n, n) array of win probabilities
        - positions: (n, 6) array of position definitions
    """
    print(f"Generating bearoff table for {max_checkers} checkers...")

    # Generate positions
    positions = generate_all_positions_array(max_checkers)
    n = positions.shape[0]
    print(f"  Positions: {n:,}")
    print(f"  Table size: {n*n:,} entries ({n*n*4/1e6:.1f} MB)")

    # Compute total checkers for each position
    totals = positions.sum(axis=1)

    # Precompute transitions
    print("  Precomputing transitions...")
    start = time.time()
    transitions = precompute_transitions(positions)
    print(f"    Done in {time.time()-start:.1f}s")

    # Initialize table
    table = np.zeros((n, n), dtype=np.float32)

    # Set terminal states
    # X borne off (total=0): X wins -> 1.0
    # O borne off (total=0): O wins -> 0.0
    x_done = totals == 0
    o_done = totals == 0

    for i in range(n):
        if x_done[i]:
            table[i, :] = 1.0
        for j in range(n):
            if o_done[j] and not x_done[i]:
                table[i, j] = 0.0

    # Convert to JAX
    table_jax = jnp.array(table)
    transitions_jax = jnp.array(transitions)
    n_dice = DICE_OUTCOMES_NP.shape[0]

    # Value iteration
    print("  Running value iteration...")
    start = time.time()

    for iteration in range(max_iterations):
        # Compute new values
        new_table_jax = value_iteration_step(table_jax, transitions_jax, n_dice)

        # Preserve terminal states
        for i in range(n):
            if x_done[i]:
                new_table_jax = new_table_jax.at[i, :].set(1.0)
        for j in range(n):
            if o_done[j]:
                # Only set where x is not done
                mask = ~x_done
                new_table_jax = new_table_jax.at[mask, j].set(0.0)

        # Check convergence
        diff = jnp.max(jnp.abs(new_table_jax - table_jax))
        diff = float(diff)

        table_jax = new_table_jax

        if iteration % 10 == 0 or diff < tolerance:
            elapsed = time.time() - start
            print(f"    Iter {iteration+1}: max_diff={diff:.2e}, elapsed={elapsed:.1f}s")

        if diff < tolerance:
            print(f"  Converged after {iteration+1} iterations")
            break

    elapsed = time.time() - start
    print(f"  Total iteration time: {elapsed:.1f}s")

    # Convert back to numpy
    table = np.array(table_jax, dtype=dtype)

    return table, positions


def benchmark_generation(max_checkers: int = 5):
    """Benchmark table generation for a given checker count."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {max_checkers} checkers")
    print(f"{'='*60}")

    start = time.time()
    table, positions = generate_bearoff_table_jax(max_checkers)
    elapsed = time.time() - start

    n = positions.shape[0]
    print(f"\nResults:")
    print(f"  Positions: {n:,}")
    print(f"  Table entries: {n*n:,}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Entries/sec: {n*n/elapsed:,.0f}")

    # Estimate for full 15 checkers
    n_full = TOTAL_ONE_SIDED_POSITIONS  # 54,264
    entries_full = n_full * n_full  # ~2.94B

    # Assuming linear scaling (conservative)
    time_estimate = elapsed * (entries_full / (n*n))
    print(f"\nEstimate for 15 checkers:")
    print(f"  Positions: {n_full:,}")
    print(f"  Entries: {entries_full:,}")
    print(f"  Estimated time: {time_estimate/3600:.1f} hours")

    # Sample values
    print(f"\nSample values:")
    for i in range(min(5, n)):
        for j in range(min(5, n)):
            print(f"  V({tuple(positions[i])}, {tuple(positions[j])}) = {table[i,j]:.4f}")

    return table, positions, elapsed


if __name__ == "__main__":
    # Run benchmark
    benchmark_generation(5)
