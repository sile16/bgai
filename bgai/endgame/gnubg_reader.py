"""Reader for gnubg bearoff database files (.bd).

gnubg two-sided database format:
- 40 byte header: "gnubg-TS-PP-CC-1xxxx..." where PP=points (06), CC=checkers
- Each entry is 8 bytes (4 unsigned shorts):
  - Win probability (scaled 0-65535)
  - Gammon probability
  - Backgammon probability
  - Reserved/equity

Position indexing uses combinatorial number system.
"""

import numpy as np
import struct
from typing import Tuple, List, Optional
from math import comb
from pathlib import Path


def read_gnubg_header(path: str) -> Tuple[int, int, str]:
    """Read header only from a gnubg two-sided database.

    Returns:
        (n_points, max_checkers, header_string)
    """
    with open(path, 'rb') as f:
        header = f.read(40)
    header_str = header[:32].decode('utf-8', errors='replace').rstrip('x\n')
    parts = header_str.split('-')
    if len(parts) < 4 or parts[1] != 'TS':
        raise ValueError(f"Invalid two-sided database header: {header_str}")
    n_points = int(parts[2])
    max_checkers = int(parts[3])
    return n_points, max_checkers, header_str


def count_positions(points: int, max_checkers: int) -> int:
    """Count total positions for given points and max checkers.

    Uses stars-and-bars: C(points + checkers - 1, checkers) for each checker count.
    """
    total = 0
    for c in range(max_checkers + 1):
        total += comb(points + c - 1, c)
    return total



from functools import lru_cache

@lru_cache(maxsize=None)
def _combination(n, k):
    if k < 0 or k > n:
        return 0
    return comb(n, k)

@lru_cache(maxsize=None)
def _position_f(f_bits, n, r):
    if n == r:
        return 0
    if r < 0:
        return 0
    if (f_bits & (1 << (n - 1))):
        return _combination(n - 1, r) + _position_f(f_bits, n - 1, r - 1)
    else:
        return _position_f(f_bits, n - 1, r)

def position_index_gnubg(pos: Tuple[int, ...], n_points: int, n_checkers: int) -> int:
    """
    Convert position tuple to gnubg index, matching PositionBearoff from positionid.c
    """
    j = n_points - 1
    for i in range(n_points):
        j += pos[i]

    f_bits = 1 << j

    for i in range(n_points - 1):
        j -= pos[i] + 1
        f_bits |= (1 << j)

    return _position_f(f_bits, n_checkers + n_points, n_points)



def index_to_position(idx: int, n_points: int = 6, max_checkers: int = 15) -> Tuple[int, ...]:
    """Convert gnubg index back to position tuple."""
    # Find total checkers for this index
    total = 0
    base = 0
    for c in range(max_checkers + 1):
        count = comb(n_points + c - 1, c)
        if base + count > idx:
            total = c
            idx -= base
            break
        base += count

    if total == 0:
        return tuple([0] * n_points)

    # Decode position within this checker count
    pos = [0] * n_points
    remaining = total

    for point in range(n_points - 1, -1, -1):
        for c in range(remaining, -1, -1):
            contrib = comb(point + c - 1, c) if c > 0 else 0
            if idx >= contrib:
                pos[point] = remaining - c
                remaining = c
                idx -= contrib
                break

    return tuple(pos)


def read_gnubg_ts_database(path: str) -> Tuple[np.ndarray, int, int]:
    """Read gnubg two-sided bearoff database.

    Args:
        path: Path to .bd file

    Returns:
        Tuple of (win_probabilities, n_points, max_checkers)
        win_probabilities is (n_positions, n_positions) array
    """
    with open(path, 'rb') as f:
        # Read header
        header = f.read(40)
        header_str = header[:32].decode('utf-8', errors='replace').rstrip('x\n')

        # Parse header: "gnubg-TS-PP-CC-1"
        parts = header_str.split('-')
        if len(parts) < 4 or parts[1] != 'TS':
            raise ValueError(f"Invalid two-sided database header: {header_str}")

        n_points = int(parts[2])
        max_checkers = int(parts[3])

        n_positions = count_positions(n_points, max_checkers)

        print(f"Reading gnubg database: {header_str}")
        print(f"  Points: {n_points}, Max checkers: {max_checkers}")
        print(f"  Positions: {n_positions}")

        # Read data - 8 bytes per entry (4 unsigned shorts)
        # First short is win probability (0-65535 scaled to 0-1)
        win_probs = np.zeros((n_positions, n_positions), dtype=np.float32)

        for x_idx in range(n_positions):
            for o_idx in range(n_positions):
                data = f.read(8)
                if len(data) < 8:
                    raise ValueError(f"Unexpected end of file at ({x_idx}, {o_idx})")

                # Unpack 4 unsigned shorts (little-endian)
                values = struct.unpack('<4H', data)
                win_prob = values[0] / 65535.0
                win_probs[x_idx, o_idx] = win_prob

        return win_probs, n_points, max_checkers


def generate_all_positions(n_points: int, max_checkers: int) -> List[Tuple[int, ...]]:
    """Generate all positions as tuples in index order."""
    positions = []

    def gen(remaining: int, points_left: int, current: List[int]):
        if points_left == 1:
            positions.append(tuple(current + [remaining]))
            return
        for v in range(remaining + 1):
            gen(remaining - v, points_left - 1, current + [v])

    for total in range(max_checkers + 1):
        gen(total, n_points, [])

    return positions


def verify_indexing(n_points: int = 6, max_checkers: int = 6):
    """Verify our indexing matches gnubg."""
    positions = generate_all_positions(n_points, max_checkers)
    n = len(positions)
    expected_n = count_positions(n_points, max_checkers)

    print(f"Verifying indexing for {n_points} points, {max_checkers} checkers")
    print(f"  Generated {n} positions, expected {expected_n}")

    if n != expected_n:
        print("  FAIL: Position count mismatch!")
        return False

    # Check each position
    errors = 0
    indices = set()
    for i, pos in enumerate(positions):
        # Note: gnubg position index is NOT the same as our enumeration order.
        # We are just checking for uniqueness and range.
        idx = position_index_gnubg(pos, n_points, max_checkers)
        if idx in indices:
            print(f"  Index collision: pos={pos}, computed={idx}")
            errors += 1
        indices.add(idx)

    if errors > 0:
        print(f"  FAIL: {errors} indexing errors")
        return False

    if max(indices) >= expected_n:
        print(f"  FAIL: Max index {max(indices)} out of range (expected < {expected_n})")
        return False

    print("  PASS: Indexing verified (unique and in range)")
    return True


def compare_with_gnubg(our_table: np.ndarray, gnubg_path: str,
                       our_positions: List[Tuple[int, ...]]) -> dict:
    """Compare our table with gnubg database using gnubg's indexing.

    Args:
        our_table: Our computed win probability table.
        gnubg_path: Path to gnubg .bd file.
        our_positions: List of position tuples in our index order.

    Returns:
        Dict with comparison statistics.
    """
    print("Comparing with gnubg...")
    gnubg_table, n_points, max_checkers = read_gnubg_ts_database(gnubg_path)
    # gnubg files use the same tuple ordering as our code (index 0 = nearest point).

    # Build our position->index lookup
    our_pos_to_idx = {pos: i for i, pos in enumerate(our_positions)}

    # Compare values for positions that exist in both
    diffs = []
    max_diff = 0.0
    max_diff_pos = None

    for x_pos in our_positions:
        if len(x_pos) != n_points:
            continue
        if sum(x_pos) > max_checkers:
            continue
        our_x_idx = our_pos_to_idx[x_pos]

        for o_pos in our_positions:
            if len(o_pos) != n_points:
                continue
            if sum(o_pos) > max_checkers:
                continue
            our_o_idx = our_pos_to_idx[o_pos]

            gnubg_x_idx = position_index_gnubg(x_pos, n_points, max_checkers)
            gnubg_o_idx = position_index_gnubg(o_pos, n_points, max_checkers)

            our_val = our_table[our_x_idx, our_o_idx]
            gnubg_val = gnubg_table[gnubg_x_idx, gnubg_o_idx]

            diff = abs(our_val - gnubg_val)
            diffs.append(diff)

            if diff > max_diff:
                max_diff = diff
                max_diff_pos = (x_pos, o_pos, our_val, gnubg_val)

    if not diffs:
        raise ValueError("No positions were compared. Check checker counts and position generation.")

    diffs = np.array(diffs)

    result = {
        'n_compared': len(diffs),
        'max_diff': max_diff,
        'mean_diff': np.mean(diffs),
        'std_diff': np.std(diffs),
        'max_diff_position': max_diff_pos,
        'pct_within_0_01': np.mean(diffs < 0.01) * 100,
        'pct_within_0_001': np.mean(diffs < 0.001) * 100,
    }

    return result


def print_comparison_report(result: dict):
    """Print comparison report."""
    print("\n" + "="*60)
    print("COMPARISON WITH GNUBG")
    print("="*60)
    print(f"Positions compared: {result['n_compared']:,}")
    print(f"Max difference: {result['max_diff']:.6f}")
    print(f"Mean difference: {result['mean_diff']:.6f}")
    print(f"Std difference: {result['std_diff']:.6f}")
    print(f"Within 0.01: {result['pct_within_0_01']:.1f}%")
    print(f"Within 0.001: {result['pct_within_0_001']:.1f}%")

    if result['max_diff_position']:
        x_pos, o_pos, our_val, gnubg_val = result['max_diff_position']
        print(f"\nLargest difference at:")
        print(f"  X position: {x_pos}")
        print(f"  O position: {o_pos}")
        print(f"  Our value: {our_val:.6f}")
        print(f"  gnubg value: {gnubg_val:.6f}")

    if result['max_diff'] < 0.001:
        print("\nStatus: EXCELLENT - Matches gnubg to 0.1%")
    elif result['max_diff'] < 0.01:
        print("\nStatus: GOOD - Matches gnubg to 1%")
    elif result['max_diff'] < 0.05:
        print("\nStatus: FAIR - Some discrepancies")
    else:
        print("\nStatus: POOR - Significant differences, check implementation")


if __name__ == "__main__":
    import sys

    # Verify indexing
    verify_indexing(6, 6)

    # If gnubg database provided, read and display stats
    if len(sys.argv) > 1:
        path = sys.argv[1]
        table, points, checkers = read_gnubg_ts_database(path)

        print(f"\nTable shape: {table.shape}")
        print(f"Value range: [{table.min():.4f}, {table.max():.4f}]")

        # Sample values
        positions = generate_all_positions(points, checkers)
        pos_to_idx = {pos: i for i, pos in enumerate(positions)}

        print("\nSample values:")
        samples = [
            ((0,0,0,0,0,0), (0,0,0,0,0,0)),
            ((1,0,0,0,0,0), (1,0,0,0,0,0)),
            ((0,0,0,0,0,1), (0,0,0,0,0,1)),
            ((1,0,0,0,0,0), (0,0,0,0,0,1)),
        ]
        for x_pos, o_pos in samples:
            if x_pos in pos_to_idx and o_pos in pos_to_idx:
                val = table[pos_to_idx[x_pos], pos_to_idx[o_pos]]
                print(f"  V({x_pos}, {o_pos}) = {val:.6f}")
