"""Corrected two-sided bearoff table generator with proper move enumeration.

V(X, O) = P(X wins | X to move)

Uses the CORRECT minimax formula:
V(X, O) = 1 - E_dice[ min_{X'} V(O, X') ]

where min is over ALL legal moves (not a greedy heuristic).

Key insight: Must enumerate ALL legal moves and take the minimum
(optimal play from X's perspective minimizes opponent's value).
"""

import numpy as np
from typing import Tuple, List, Dict, Set
import time
from tqdm import tqdm


MAX_CHECKERS = 15
NUM_POINTS = 6

# All 21 unique dice outcomes with their probabilities
# Doubles (1/36 each), non-doubles (2/36 each)
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
DICE_PROBS = np.array(DICE_PROBS)


def get_legal_moves_for_die(pos: Tuple[int, ...], die: int) -> List[Tuple[int, ...]]:
    """
    Generates all legal positions resulting from playing a single die.
    Assumes all checkers are in the home board (a bearoff position).

    Returns a list of all possible resulting positions.
    """
    moves = []
    pos_list = list(pos)

    # Rule 1: Move a checker from a point to a lower point.
    for p in range(die, NUM_POINTS):
        if pos_list[p] > 0:
            new_pos = pos_list[:]
            new_pos[p] -= 1
            new_pos[p - die] += 1
            moves.append(tuple(new_pos))

    # Rule 2: Bear off a checker from the point corresponding to the die.
    if pos_list[die - 1] > 0:
        new_pos = pos_list[:]
        new_pos[die - 1] -= 1
        moves.append(tuple(new_pos))

    # Rule 3: If no move under Rule 1 or 2 is possible, you can bear off from a
    # higher point if the die roll is greater than your highest point.
    if not moves:
        highest_occupied = -1
        for i in range(NUM_POINTS - 1, -1, -1):
            if pos_list[i] > 0:
                highest_occupied = i
                break

        if highest_occupied != -1 and die > highest_occupied + 1:
            new_pos = pos_list[:]
            new_pos[highest_occupied] -= 1
            moves.append(tuple(new_pos))

    return list(set(moves))


def apply_dice(pos: Tuple[int, ...], dice: Tuple[int, int]) -> List[Tuple[int, ...]]:
    """
    Generates all legal final positions for a dice roll (d1, d2).
    Handles non-doubles and doubles, and the rule that you must play as
    much of the roll as possible.

    Returns a list of ALL possible resulting positions.
    """
    d1, d2 = dice
    if d1 == d2:  # Doubles
        # Apply the die up to 4 times
        positions: Set[Tuple[int, ...]] = {pos}
        for _ in range(4):
            next_positions: Set[Tuple[int, ...]] = set()
            for p in positions:
                moves = get_legal_moves_for_die(p, d1)
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
    for p1 in get_legal_moves_for_die(pos, d1):
        moves_d2 = get_legal_moves_for_die(p1, d2)
        if moves_d2:
            two_move_plays.update(moves_d2)
        else:
            one_move_plays.add(p1)

    # Path 2: d2 then d1
    for p2 in get_legal_moves_for_die(pos, d2):
        moves_d1 = get_legal_moves_for_die(p2, d1)
        if moves_d1:
            two_move_plays.update(moves_d1)
        else:
            one_move_plays.add(p2)

    if two_move_plays:
        return list(two_move_plays)

    if one_move_plays:
        return list(one_move_plays)

    return [pos]  # No moves possible


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
    """Generate bearoff table with correct minimax DP.

    Uses the CORRECT formula:
    V(X, O) = 1 - E_dice[ min_{X'} V(O, X') ]

    where min is over ALL legal moves (optimal play).

    Returns:
        (table, positions) where table[x_idx, o_idx] = P(X wins | X to move)
    """
    print(f"Generating bearoff table for up to {max_checkers} checkers...")

    positions = generate_positions(max_checkers)
    n = len(positions)
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}

    print(f"  Positions: {n:,}")
    print(f"  Table size: {n*n:,}")

    # Precompute position totals
    totals = np.array([sum(p) for p in positions], dtype=np.int32)

    # Precompute ALL transitions: trans[pos_idx][dice_idx] = list of new_pos_indices
    # This is a list of lists because each dice roll can have multiple legal moves
    print("  Precomputing transitions (all legal moves)...")
    n_dice = len(DICE_OUTCOMES)
    trans: List[List[List[int]]] = []  # trans[pos_idx][dice_idx] = [new_pos_idx, ...]

    for i, pos in enumerate(tqdm(positions)):
        pos_trans = []
        for j, dice in enumerate(DICE_OUTCOMES):
            new_positions = apply_dice(pos, dice)
            new_indices = [pos_to_idx[new_pos] for new_pos in new_positions]
            pos_trans.append(new_indices)
        trans.append(pos_trans)

    # Initialize value table
    # V(X, O) = P(X wins | X to move)
    table = np.zeros((n, n), dtype=np.float64)

    # Terminal conditions:
    # If X has 0 checkers (borne off), X has won: V = 1
    # If O has 0 checkers but X hasn't won yet, X loses: V = 0

    done_mask = totals == 0  # Position is borne off

    # Initialize: if X done, V=1; if O done and X not done, V=0
    for x_idx in range(n):
        if done_mask[x_idx]:
            table[x_idx, :] = 1.0
        else:
            for o_idx in range(n):
                if done_mask[o_idx]:
                    table[x_idx, o_idx] = 0.0

    # Value iteration with CORRECT minimax formula:
    # V(X, O) = 1 - E_dice[ min_{X' in legal_moves(X, dice)} V(O, X') ]
    #
    # X picks the move that minimizes opponent's value (= maximizes own value)

    print("  Running value iteration (minimax)...")
    start = time.time()

    for iteration in range(max_iterations):
        old_table = table.copy()

        # For non-terminal positions only
        for x_idx in tqdm(range(n), desc=f"Iter {iteration+1}", disable=iteration > 0):
            if done_mask[x_idx]:
                continue

            x_pos = positions[x_idx]

            for o_idx in range(n):
                if done_mask[o_idx]:
                    continue

                # V(X, O) = 1 - E_dice[ min_{X'} V(O, X') ]
                expected_min_opp_value = 0.0

                for dice_idx, prob in enumerate(DICE_PROBS):
                    # Get all legal moves for this dice roll
                    x_new_indices = trans[x_idx][dice_idx]

                    # Find the minimum opponent value over all legal moves
                    # (optimal play: X picks move that minimizes O's winning chance)
                    min_opp_value = 1.0
                    for x_new_idx in x_new_indices:
                        if done_mask[x_new_idx]:
                            # X borne off -> X wins -> O's value from this state is 0
                            opp_value = 0.0
                        else:
                            # V(O, X') from old table
                            opp_value = old_table[o_idx, x_new_idx]

                        if opp_value < min_opp_value:
                            min_opp_value = opp_value

                    expected_min_opp_value += prob * min_opp_value

                table[x_idx, o_idx] = 1.0 - expected_min_opp_value

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
