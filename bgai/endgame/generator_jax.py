"""JAX-accelerated two-sided bearoff table generator.

Uses vectorized operations for fast value iteration.
Correct formula: V(X, O) = P(X wins | X to move)

Supports three modes:
1. GPU vectorized: Small tables that fit entirely on GPU
2. GPU batched: Medium tables with batched GPU processing
3. CPU JAX: Large tables using JAX on CPU (JIT-compiled, faster than numpy)
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, List
import time
from tqdm import tqdm

from .indexing import (
    TOTAL_ONE_SIDED_POSITIONS,
    MAX_CHECKERS,
    NUM_POINTS,
)


# All 36 dice outcomes (enumerate all for correct probability)
ALL_DICE = [(d1, d2) for d1 in range(1, 7) for d2 in range(1, 7)]
DICE_PROB = 1.0 / 36.0


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


def apply_single_die_np(pos: np.ndarray, die: int) -> np.ndarray:
    """Apply single die move optimally in bearoff (numpy)."""
    pos = pos.copy()
    total = pos.sum()

    if total == 0:
        return pos

    # Can bear off from exact point?
    if pos[die - 1] > 0:
        pos[die - 1] -= 1
        return pos

    # Find highest occupied point
    highest = -1
    for i in range(5, -1, -1):
        if pos[i] > 0:
            highest = i
            break

    if highest < 0:
        return pos

    # Die > highest point + 1? Bear off from highest
    if die > highest + 1:
        pos[highest] -= 1
        return pos

    # Move from highest point that can legally move
    for src in range(5, -1, -1):
        dst = src - die
        if pos[src] > 0 and dst >= 0:
            pos[src] -= 1
            pos[dst] += 1
            return pos

    return pos


def apply_dice_np(pos: np.ndarray, d1: int, d2: int) -> np.ndarray:
    """Apply a dice roll (d1, d2) optimally (numpy)."""
    if d1 == d2:
        # Doubles: 4 moves
        for _ in range(4):
            pos = apply_single_die_np(pos, d1)
    else:
        # Try both orderings, pick better result
        pos1 = apply_single_die_np(pos.copy(), d1)
        pos1 = apply_single_die_np(pos1, d2)

        pos2 = apply_single_die_np(pos.copy(), d2)
        pos2 = apply_single_die_np(pos2, d1)

        # Prefer fewer checkers remaining
        if pos1.sum() <= pos2.sum():
            pos = pos1
        else:
            pos = pos2

    return pos


def precompute_transitions(positions: List[Tuple[int, ...]],
                           pos_to_idx: dict) -> np.ndarray:
    """Precompute transition table.

    Returns:
        trans[pos_idx, dice_idx] = new_pos_idx
    """
    n = len(positions)
    n_dice = len(ALL_DICE)
    trans = np.zeros((n, n_dice), dtype=np.int32)

    for i, pos in enumerate(tqdm(positions, desc="Precomputing transitions")):
        pos_arr = np.array(pos, dtype=np.int32)
        for j, (d1, d2) in enumerate(ALL_DICE):
            new_pos = apply_dice_np(pos_arr, d1, d2)
            trans[i, j] = pos_to_idx[tuple(new_pos)]

    return trans


def generate_bearoff_table_jax(max_checkers: int = MAX_CHECKERS,
                                max_iterations: int = 100,
                                tolerance: float = 1e-8,
                                batch_size: int = 1000) -> Tuple[np.ndarray, List[Tuple]]:
    """Generate bearoff table with JAX-accelerated value iteration.

    V(X, O) = P(X wins | X to move)

    The update rule:
    V(X, O) = E_d1[ I(X'=0)*1 + I(X'≠0)*E_d2[ I(O'=0)*0 + I(O'≠0)*V(X', O') ] ]

    Uses batched processing to handle large tables without OOM.

    Returns:
        (table, positions) where table[x_idx, o_idx] = P(X wins | X to move)
    """
    print(f"Generating bearoff table for {max_checkers} checkers (JAX-accelerated)...")

    # Generate positions
    positions = generate_all_positions(max_checkers)
    n = len(positions)
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}

    print(f"  Positions: {n:,}")
    print(f"  Table entries: {n*n:,} ({n*n*4/1e9:.2f} GB)")

    # Compute totals for each position
    totals = np.array([sum(p) for p in positions], dtype=np.int32)

    # Masks for terminal conditions
    x_done_mask = totals == 0  # X has borne off

    # Precompute transitions
    print("  Precomputing transitions...")
    start = time.time()
    trans = precompute_transitions(positions, pos_to_idx)
    print(f"    Done in {time.time()-start:.1f}s")

    # Convert to JAX arrays
    trans_jax = jnp.array(trans)
    x_done_jax = jnp.array(x_done_mask)

    # Initialize table
    table = np.zeros((n, n), dtype=np.float32)

    # Set terminal states
    # X borne off: X wins -> 1.0
    table[x_done_mask, :] = 1.0
    # O borne off (and X not): X loses -> 0.0 (already 0)

    n_dice = trans.shape[1]

    # Check if we can use fully vectorized approach (small tables)
    memory_estimate_gb = n * n * n_dice * 4 / 1e9
    use_batched = memory_estimate_gb > 8.0  # Use batched if > 8GB intermediate
    use_cpu_table = n * n * 4 / 1e9 > 4.0  # Use CPU if table > 4GB

    # Only convert to JAX if we're using GPU
    if not use_cpu_table:
        table_jax = jnp.array(table)
        # Precompute o_done_after for all O positions on GPU
        o_done_after = x_done_jax[trans_jax]  # (n, n_dice)
        x_done_after = o_done_after  # Same array since both use same mask
    else:
        table_jax = None  # Will use table_cpu instead

    if use_batched:
        print(f"  Using batched processing (batch_size={batch_size})")

        # Precompute numpy arrays for masks
        trans_np = np.array(trans)
        x_done_np = np.array(x_done_mask)
        x_done_after_np = x_done_np[trans_np]  # (n, n_dice)
        mask_np = ~x_done_after_np  # True where O doesn't bear off after their move
        weighted_mask_np = DICE_PROB * mask_np.astype(np.float32)  # (n, 36)

        if use_cpu_table:
            print(f"  Using JAX CPU backend with batching for table ({n*n*4/1e9:.2f} GB)")

            # Get CPU device for explicit placement
            cpu_device = jax.devices('cpu')[0]

            # Use numpy for the table (too large for JAX arrays to copy efficiently)
            # JAX JIT will be used for computing batches

            # Create JIT-compiled batch computation function
            @partial(jax.jit, device=cpu_device)
            def compute_inner_sum_batch(table: jnp.ndarray, trans_batch: jnp.ndarray,
                                         weighted_mask: jnp.ndarray) -> jnp.ndarray:
                """Compute inner sum for a batch of x' positions.

                Args:
                    table: Full table (n, n)
                    trans_batch: Transitions for batch positions (batch, n_dice)
                    weighted_mask: Weighted mask (n, n_dice)

                Returns:
                    inner_sum for batch: (batch, n)
                """
                # For each x' in batch, gather table[x', trans[o, d2]] for all o, d2
                # table[trans_batch] gives (batch, n_dice) indices -> (batch, n_dice, n) doesn't work
                # Instead: table_rows[b] = table[trans_batch[b]] for each b

                def compute_row(x_prime_trans):
                    # x_prime_trans: (n_dice,) indices
                    # table[x_prime_trans] -> (n_dice, n)
                    table_gathered = table[x_prime_trans]  # (n_dice, n)
                    # inner_sum[o] = sum_d2 weighted_mask[o, d2] * table_gathered[d2, o]
                    # = sum_d2 weighted_mask[o, d2] * table_gathered.T[o, d2]
                    return jnp.sum(weighted_mask * table_gathered.T, axis=1)  # (n,)

                return jax.vmap(compute_row)(trans_batch)  # (batch, n)

            @partial(jax.jit, device=cpu_device)
            def compute_new_values_batch(inner_sum: jnp.ndarray, trans_batch: jnp.ndarray,
                                          x_done_batch: jnp.ndarray, x_done_after_batch: jnp.ndarray,
                                          x_done_all: jnp.ndarray) -> jnp.ndarray:
                """Compute new table values for a batch of x positions.

                Args:
                    inner_sum: Precomputed inner sums (n, n)
                    trans_batch: Transitions for batch (batch, n_dice)
                    x_done_batch: X done mask for batch (batch,)
                    x_done_after_batch: X done after move for batch (batch, n_dice)
                    x_done_all: X done mask for all positions (n,)

                Returns:
                    New values for batch: (batch, n)
                """
                # inner_sum_for_x[b, d1, o] = inner_sum[trans_batch[b, d1], o]
                inner_sum_for_x = inner_sum[trans_batch]  # (batch, n_dice, n)

                # Compute outer sum
                outer_terms = jnp.where(
                    x_done_after_batch[:, :, None],  # (batch, n_dice, 1)
                    1.0,
                    inner_sum_for_x  # (batch, n_dice, n)
                )

                new_batch = jnp.sum(DICE_PROB * outer_terms, axis=1)  # (batch, n)

                # Preserve terminal states
                new_batch = jnp.where(x_done_batch[:, None], 1.0, new_batch)
                new_batch = jnp.where(x_done_all[None, :] & ~x_done_batch[:, None], 0.0, new_batch)

                return new_batch

            # Prepare arrays on CPU - keep table as numpy for memory efficiency
            trans_cpu = jax.device_put(trans, cpu_device)
            x_done_cpu = jax.device_put(x_done_mask, cpu_device)
            weighted_mask_cpu = jax.device_put(weighted_mask_np, cpu_device)
            x_done_after_cpu = x_done_cpu[trans_cpu]  # (n, n_dice)

            # Keep tables as numpy arrays for explicit memory control
            table_np = table.astype(np.float32)  # Current table
            new_table_np = np.zeros_like(table_np)  # Will be filled in-place
            inner_sum_np = np.zeros((n, n), dtype=np.float32)  # Reusable buffer

            # Prepare batch indices - smaller batches for 15 checker case
            cpu_batch_size = min(batch_size, 1000)
            batch_starts = list(range(0, n, cpu_batch_size))

            # Value iteration on CPU with batching
            print(f"  Running value iteration (JAX CPU batched, batch_size={cpu_batch_size})...")
            start = time.time()

            import gc

            for iteration in range(max_iterations):
                # Move current table to CPU JAX (will be read-only this iteration)
                table_cpu = jax.device_put(table_np, cpu_device)

                # Step 1: Compute inner_sum for all x' positions (in batches)
                for b_start in batch_starts:
                    b_end = min(b_start + cpu_batch_size, n)
                    trans_batch = trans_cpu[b_start:b_end]
                    inner_sum_batch = compute_inner_sum_batch(table_cpu, trans_batch, weighted_mask_cpu)
                    # Block until ready and copy to numpy
                    inner_sum_np[b_start:b_end] = np.array(inner_sum_batch)

                # Move inner_sum to JAX for step 2
                inner_sum_cpu = jax.device_put(inner_sum_np, cpu_device)

                # Step 2: Compute new values for all x positions (in batches)
                max_diff = 0.0
                for b_start in batch_starts:
                    b_end = min(b_start + cpu_batch_size, n)
                    trans_batch = trans_cpu[b_start:b_end]
                    x_done_batch = x_done_cpu[b_start:b_end]
                    x_done_after_batch = x_done_after_cpu[b_start:b_end]

                    new_batch = compute_new_values_batch(
                        inner_sum_cpu, trans_batch, x_done_batch, x_done_after_batch, x_done_cpu
                    )
                    # Block until ready and copy to numpy
                    new_batch_np = np.array(new_batch)

                    # Compute local max diff
                    local_diff = np.max(np.abs(new_batch_np - table_np[b_start:b_end]))
                    max_diff = max(max_diff, local_diff)

                    # Write directly to new table
                    new_table_np[b_start:b_end] = new_batch_np

                # Swap tables (just swap references, no copy)
                table_np, new_table_np = new_table_np, table_np

                # Clean up JAX references and force garbage collection
                del table_cpu, inner_sum_cpu
                gc.collect()

                if iteration % 5 == 0 or max_diff < tolerance:
                    elapsed = time.time() - start
                    print(f"    Iter {iteration+1}: max_diff={max_diff:.2e}, elapsed={elapsed:.1f}s")

                if max_diff < tolerance:
                    print(f"  Converged after {iteration+1} iterations")
                    break

            elapsed = time.time() - start
            print(f"  Total value iteration time: {elapsed:.1f}s")

            # Return final table
            return table_np, positions

        else:
            # GPU batched path
            weighted_mask_jax = jnp.array(weighted_mask_np)

            @jax.jit
            def compute_response_batch(xp_indices: jnp.ndarray, table: jnp.ndarray) -> jnp.ndarray:
                """Compute response for a batch of x' positions."""
                def compute_row(x_prime):
                    table_row_gathered = table[x_prime, trans_jax]  # (n_o, n_dice)
                    return jnp.sum(weighted_mask_jax * table_row_gathered, axis=1)  # (n_o,)

                return jax.vmap(compute_row)(xp_indices)

            @jax.jit
            def compute_batch_from_response(batch_indices: jnp.ndarray,
                                             response: jnp.ndarray) -> jnp.ndarray:
                """Compute new values using precomputed response matrix."""
                x_trans_batch = trans_jax[batch_indices]
                x_done_batch = x_done_jax[batch_indices]
                x_done_after_batch = o_done_after[batch_indices]

                response_at_xp = response[x_trans_batch]

                outer_terms = jnp.where(
                    x_done_after_batch[:, :, None],
                    1.0,
                    response_at_xp
                )

                new_batch = jnp.sum(DICE_PROB * outer_terms, axis=1)
                new_batch = jnp.where(x_done_batch[:, None], 1.0, new_batch)
                new_batch = jnp.where(x_done_jax[None, :] & ~x_done_batch[:, None], 0.0, new_batch)

                return new_batch

            # Precompute batch indices
            batch_indices_list = []
            for x_start in range(0, n, batch_size):
                x_end = min(x_start + batch_size, n)
                batch_indices_list.append(jnp.arange(x_start, x_end))

            response_batch_size = min(batch_size, 500)
            response_batch_indices = []
            for start in range(0, n, response_batch_size):
                end = min(start + response_batch_size, n)
                response_batch_indices.append(jnp.arange(start, end))

            print("  Running value iteration (JAX GPU batched)...")
            start = time.time()

            for iteration in range(max_iterations):
                # GPU-only path: compute response in batches then concatenate
                response_list = []
                for batch_idx in response_batch_indices:
                    response_batch = compute_response_batch(batch_idx, table_jax)
                    response_list.append(response_batch)
                response = jnp.concatenate(response_list, axis=0)

                new_table_list = []
                for batch_indices in batch_indices_list:
                    batch_result = compute_batch_from_response(batch_indices, response)
                    new_table_list.append(batch_result)

                new_table_jax = jnp.concatenate(new_table_list, axis=0)

                diff = float(jnp.max(jnp.abs(new_table_jax - table_jax)))
                table_jax = new_table_jax

                if iteration % 5 == 0 or diff < tolerance:
                    elapsed = time.time() - start
                    print(f"    Iter {iteration+1}: max_diff={diff:.2e}, elapsed={elapsed:.1f}s")

                if diff < tolerance:
                    print(f"  Converged after {iteration+1} iterations")
                    break

            elapsed = time.time() - start
            print(f"  Total value iteration time: {elapsed:.1f}s")

            # Return from GPU batched path
            return np.array(table_jax, dtype=np.float32), positions

    else:
        # Fully vectorized approach for smaller tables
        @jax.jit
        def value_iteration_step_vectorized(table: jnp.ndarray) -> jnp.ndarray:
            """Fully vectorized value iteration step."""
            # table_gathered[x, o, d2] = table[x, trans[o, d2]]
            table_gathered = table[:, trans_jax]  # (n, n, n_dice)

            # inner_sum[x', o] = sum_d2 P * (~o_done_after[o,d2]) * table_gathered[x', o, d2]
            mask = ~o_done_after  # (n_o, n_dice)
            inner_sum = jnp.sum(DICE_PROB * mask[None, :, :] * table_gathered, axis=2)  # (n, n)

            # inner_sum_for_x[x, d1, o] = inner_sum[trans[x, d1], o]
            inner_sum_for_x = inner_sum[trans_jax]  # (n_x, n_dice, n_o)

            # Compute outer sum
            outer_terms = jnp.where(
                x_done_after[:, :, None],  # (n_x, n_dice, 1)
                1.0,
                inner_sum_for_x  # (n_x, n_dice, n_o)
            )

            new_table = jnp.sum(DICE_PROB * outer_terms, axis=1)  # (n_x, n_o)

            # Preserve terminal states
            new_table = jnp.where(x_done_jax[:, None], 1.0, new_table)
            new_table = jnp.where(x_done_jax[None, :] & ~x_done_jax[:, None], 0.0, new_table)

            return new_table

        # Run value iteration
        print("  Running value iteration (JAX vectorized)...")
        start = time.time()

        for iteration in range(max_iterations):
            new_table_jax = value_iteration_step_vectorized(table_jax)

            diff = float(jnp.max(jnp.abs(new_table_jax - table_jax)))
            table_jax = new_table_jax

            if iteration % 5 == 0 or diff < tolerance:
                elapsed = time.time() - start
                print(f"    Iter {iteration+1}: max_diff={diff:.2e}, elapsed={elapsed:.1f}s")

            if diff < tolerance:
                print(f"  Converged after {iteration+1} iterations")
                break

    elapsed = time.time() - start
    print(f"  Total value iteration time: {elapsed:.1f}s")

    # GPU vectorized path: convert from JAX
    return np.array(table_jax, dtype=np.float32), positions


def verify_table(table: np.ndarray, positions: List[Tuple]) -> None:
    """Verify table values for known positions."""
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}

    print("\n=== Table verification ===")

    test_cases = [
        ((1,0,0,0,0,0), (1,0,0,0,0,0), None, "Both on 1pt"),
        ((0,0,0,0,0,1), (1,0,0,0,0,0), 0.75, "X on 6pt, O on 1pt"),
        ((1,0,0,0,0,0), (0,0,0,0,0,1), 1.0, "X on 1pt, O on 6pt"),
        ((0,0,0,0,0,1), (0,0,0,0,0,1), None, "Both on 6pt"),
    ]

    for x_pos, o_pos, expected, desc in test_cases:
        if x_pos not in pos_to_idx or o_pos not in pos_to_idx:
            continue
        x_idx = pos_to_idx[x_pos]
        o_idx = pos_to_idx[o_pos]
        val = table[x_idx, o_idx]
        if expected:
            status = "✓" if abs(val - expected) < 0.001 else "✗"
            print(f"  {desc}: {val:.4f} (expected {expected}) {status}")
        else:
            print(f"  {desc}: {val:.4f}")


if __name__ == "__main__":
    # Test with small number of checkers
    table, positions = generate_bearoff_table_jax(max_checkers=5)
    verify_table(table, positions)
