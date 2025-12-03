"""Corrected two-sided bearoff table generator.

V(X, O) = P(X wins | X to move)

After X moves (dice d1) to X', then O moves (dice d2) to O':
V(X, O) = E_d1,d2[ V(X', O') ]  if neither bears off
        = E_d1[ 1 ]             if X bears off (X wins)
        = E_d1[ E_d2[ 0 ] ]     if X doesn't bear off but O does

Key insight: Check for game-ending conditions at each step.
"""

import numpy as np
from typing import Tuple, List, Dict
import time
from tqdm import tqdm


MAX_CHECKERS = 15
NUM_POINTS = 6

# All 36 dice outcomes (we enumerate all, not just unique pairs)
# This makes the expected value computation simpler
ALL_DICE = [(d1, d2) for d1 in range(1, 7) for d2 in range(1, 7)]
DICE_PROB = 1 / 36  # Each outcome equally likely


def apply_single_die(pos: np.ndarray, die: int) -> np.ndarray:
    """Apply single die optimally in bearoff."""
    pos = pos.copy()
    total = pos.sum()

    if total == 0:
        return pos

    # Indices: 0=1-point, 5=6-point
    # Die 1 bears off from index 0, die 6 bears off from index 5

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


def apply_dice(pos: np.ndarray, d1: int, d2: int) -> np.ndarray:
    """Apply a dice roll (d1, d2) optimally."""
    if d1 == d2:
        # Doubles: 4 moves
        for _ in range(4):
            pos = apply_single_die(pos, d1)
    else:
        # Try both orderings, pick better result
        pos1 = apply_single_die(pos.copy(), d1)
        pos1 = apply_single_die(pos1, d2)

        pos2 = apply_single_die(pos.copy(), d2)
        pos2 = apply_single_die(pos2, d1)

        # Prefer fewer checkers remaining
        if pos1.sum() <= pos2.sum():
            pos = pos1
        else:
            pos = pos2

    return pos


def generate_positions(max_checkers: int = MAX_CHECKERS) -> List[Tuple[int, ...]]:
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


def generate_bearoff_table(max_checkers: int = MAX_CHECKERS,
                           max_iterations: int = 200,
                           tolerance: float = 1e-9) -> Tuple[np.ndarray, List[Tuple]]:
    """Generate bearoff table with correct DP.

    Returns:
        (table, positions) where table[x_idx, o_idx] = P(X wins | X to move)
    """
    print(f"Generating bearoff table for up to {max_checkers} checkers...")

    positions = generate_positions(max_checkers)
    n = len(positions)
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}

    print(f"  Positions: {n:,}")
    print(f"  Table size: {n*n:,}")

    # Precompute position totals and transitions
    totals = np.array([sum(p) for p in positions], dtype=np.int32)
    positions_np = np.array(positions, dtype=np.int32)

    # Precompute all transitions: trans[pos_idx, dice_idx] = new_pos_idx
    print("  Precomputing transitions...")
    n_dice = len(ALL_DICE)
    trans = np.zeros((n, n_dice), dtype=np.int32)

    for i, pos in enumerate(tqdm(positions)):
        pos_arr = np.array(pos, dtype=np.int32)
        for j, (d1, d2) in enumerate(ALL_DICE):
            new_pos = apply_dice(pos_arr, d1, d2)
            trans[i, j] = pos_to_idx[tuple(new_pos)]

    # Initialize value table
    # V(X, O) = P(X wins | X to move)
    table = np.zeros((n, n), dtype=np.float64)

    # Terminal conditions:
    # If X has 0 checkers (borne off), X has won: V = 1
    # If O has 0 checkers but X hasn't won yet, X loses: V = 0

    x_done_mask = totals == 0  # X has borne off
    o_done_mask = totals == 0  # O has borne off

    # Initialize: if X done, V=1; if O done and X not done, V=0
    for x_idx in range(n):
        if x_done_mask[x_idx]:
            table[x_idx, :] = 1.0
        else:
            for o_idx in range(n):
                if o_done_mask[o_idx]:
                    table[x_idx, o_idx] = 0.0

    # Value iteration
    # V(X, O) = sum over X's dice d1:
    #     P(d1) * [
    #         if X bears off after d1: 1.0
    #         else: sum over O's dice d2:
    #             P(d2) * [
    #                 if O bears off after d2: 0.0
    #                 else: V(X', O')
    #             ]
    #     ]

    print("  Running value iteration...")
    start = time.time()

    for iteration in range(max_iterations):
        old_table = table.copy()

        # For non-terminal positions only
        for x_idx in tqdm(range(n), desc=f"Iter {iteration+1}", disable=iteration > 0):
            if x_done_mask[x_idx]:
                continue

            for o_idx in range(n):
                if o_done_mask[o_idx]:
                    continue

                # Compute expected value
                val = 0.0
                for d1_idx, (d1a, d1b) in enumerate(ALL_DICE):
                    x_new_idx = trans[x_idx, d1_idx]

                    # Did X bear off?
                    if x_done_mask[x_new_idx]:
                        val += DICE_PROB * 1.0
                    else:
                        # O's turn
                        inner_val = 0.0
                        for d2_idx in range(n_dice):
                            o_new_idx = trans[o_idx, d2_idx]

                            # Did O bear off?
                            if o_done_mask[o_new_idx]:
                                inner_val += DICE_PROB * 0.0
                            else:
                                inner_val += DICE_PROB * old_table[x_new_idx, o_new_idx]

                        val += DICE_PROB * inner_val

                table[x_idx, o_idx] = val

        # Check convergence
        diff = np.max(np.abs(table - old_table))
        elapsed = time.time() - start

        if iteration % 5 == 0 or diff < tolerance:
            print(f"    Iter {iteration+1}: max_diff={diff:.2e}, elapsed={elapsed:.1f}s")

        if diff < tolerance:
            print(f"  Converged after {iteration+1} iterations")
            break

    return table.astype(np.float32), positions


def verify_against_gnubg(table: np.ndarray, positions: List[Tuple], n_tests: int = 20):
    """Verify our table against gnubg."""
    import gnubg

    print(f"\nVerifying against gnubg ({n_tests} tests)...")

    pos_to_idx = {pos: i for i, pos in enumerate(positions)}
    errors = []

    # Test specific positions
    test_cases = [
        ((1,0,0,0,0,0), (0,0,0,0,0,1)),  # X on 1pt, O on 6pt
        ((0,0,0,0,0,1), (1,0,0,0,0,0)),  # X on 6pt, O on 1pt
        ((1,0,0,0,0,0), (1,0,0,0,0,0)),  # Both on 1pt
        ((0,0,1,0,0,0), (0,0,1,0,0,0)),  # Both on 3pt
        ((2,0,0,0,0,0), (2,0,0,0,0,0)),  # 2 each on 1pt
        ((1,1,0,0,0,0), (1,1,0,0,0,0)),  # 1 on 1pt, 1 on 2pt each
    ]

    for x_pos, o_pos in test_cases:
        if x_pos not in pos_to_idx or o_pos not in pos_to_idx:
            continue

        x_idx = pos_to_idx[x_pos]
        o_idx = pos_to_idx[o_pos]
        our_val = table[x_idx, o_idx]

        # gnubg format: [opponent, current_player]
        # board[1][i] = current player (X) checkers at X's point i
        # board[0][i] = opponent (O) checkers at O's point i (from O's perspective!)
        board = [[0]*25, [0]*25]
        for i, cnt in enumerate(x_pos):
            if cnt > 0:
                board[1][i + 1] = int(cnt)  # X on X's points 1-6
        for i, cnt in enumerate(o_pos):
            if cnt > 0:
                board[0][i + 1] = int(cnt)  # O on O's points 1-6 (from O's view)

        try:
            probs = gnubg.probabilities(board, 0)
            gnubg_val = probs[0]
            error = abs(our_val - gnubg_val)
            errors.append(error)
            status = "OK" if error < 0.01 else "MISMATCH"
            print(f"  X:{x_pos} O:{o_pos} -> ours={our_val:.4f}, gnubg={gnubg_val:.4f}, err={error:.4f} [{status}]")
        except Exception as e:
            print(f"  X:{x_pos} O:{o_pos} -> gnubg error: {e}")

    if errors:
        print(f"\n  Max error: {max(errors):.4f}")
        print(f"  Mean error: {np.mean(errors):.4f}")


if __name__ == "__main__":
    # Small test
    table, positions = generate_bearoff_table(max_checkers=3)

    print("\nSample values:")
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}
    samples = [
        ((1,0,0,0,0,0), (1,0,0,0,0,0)),
        ((0,0,0,0,0,1), (0,0,0,0,0,1)),
        ((1,0,0,0,0,0), (0,0,0,0,0,1)),
        ((0,0,0,0,0,1), (1,0,0,0,0,0)),
    ]
    for x_pos, o_pos in samples:
        if x_pos in pos_to_idx and o_pos in pos_to_idx:
            val = table[pos_to_idx[x_pos], pos_to_idx[o_pos]]
            print(f"  V({x_pos}, {o_pos}) = {val:.4f}")

    verify_against_gnubg(table, positions)
