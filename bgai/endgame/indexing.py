"""Position indexing using combinatorial number system.

Provides O(1) perfect hashing for bearoff positions.

A bearoff position is represented as a tuple of 6 integers
(checkers on points 1-6), where sum <= 15.

The indexing uses the combinatorial number system to map
each position to a unique index in [0, TOTAL_ONE_SIDED_POSITIONS).
"""

import numpy as np
from functools import lru_cache
from typing import Tuple
import jax.numpy as jnp
import jax

# Maximum checkers per player in bearoff
MAX_CHECKERS = 15
NUM_POINTS = 6

# Precompute Pascal's triangle for binomial coefficients
# We need C(n, k) for n up to MAX_CHECKERS + NUM_POINTS, k up to NUM_POINTS
_PASCAL_SIZE = MAX_CHECKERS + NUM_POINTS + 2
_PASCAL = np.zeros((_PASCAL_SIZE, _PASCAL_SIZE), dtype=np.int64)
_PASCAL[0, 0] = 1
for n in range(1, _PASCAL_SIZE):
    _PASCAL[n, 0] = 1
    for k in range(1, n + 1):
        _PASCAL[n, k] = _PASCAL[n-1, k-1] + _PASCAL[n-1, k]


def binomial(n: int, k: int) -> int:
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    return int(_PASCAL[n, k])


@lru_cache(maxsize=32)
def positions_for_checkers(num_checkers: int, num_points: int = NUM_POINTS) -> int:
    """Number of ways to place num_checkers on num_points (stars and bars).

    This equals C(num_checkers + num_points - 1, num_points - 1).
    """
    return binomial(num_checkers + num_points - 1, num_points - 1)


def total_positions(max_checkers: int = MAX_CHECKERS, num_points: int = NUM_POINTS) -> int:
    """Total number of one-sided positions for 0 to max_checkers.

    This equals C(max_checkers + num_points, num_points).
    """
    return binomial(max_checkers + num_points, num_points)


# Total one-sided positions for standard 15-checker, 6-point bearoff
TOTAL_ONE_SIDED_POSITIONS = total_positions(MAX_CHECKERS, NUM_POINTS)  # 54,264


def position_to_index(pos: Tuple[int, ...]) -> int:
    """Convert a position tuple to a unique index.

    Uses combinatorial number system for perfect hashing.

    Args:
        pos: Tuple of 6 integers (checkers on points 1-6).
             Sum must be <= MAX_CHECKERS.

    Returns:
        Unique index in [0, TOTAL_ONE_SIDED_POSITIONS).

    The encoding works by:
    1. First, index by total checker count (0-15)
    2. Within each count, use combinatorial number to index the distribution

    Actually, we use a unified encoding that maps all positions to a
    contiguous range using the combinatorial number system.
    """
    # Validate
    assert len(pos) == NUM_POINTS
    total = sum(pos)
    assert 0 <= total <= MAX_CHECKERS

    # The index is computed as:
    # For position (c1, c2, c3, c4, c5, c6) with total n checkers,
    # we compute the rank in the sequence of all distributions.
    #
    # Using stars-and-bars: position corresponds to choosing where to
    # place dividers among checkers.
    #
    # We use accumulated indexing across all checker counts.

    # Offset for positions with fewer total checkers
    offset = 0
    for k in range(total):
        offset += positions_for_checkers(k, NUM_POINTS)

    # Index within positions of this checker count
    # Using combinatorial number system
    inner_index = _pos_to_inner_index(pos, total)

    return offset + inner_index


def _pos_to_inner_index(pos: Tuple[int, ...], total: int) -> int:
    """Convert position to index within its checker-count group.

    Uses lexicographic ordering of distributions.
    """
    # We enumerate positions in lexicographic order
    # Index = sum of C(remaining + points_left - 1, points_left - 1)
    # for each "skipped" value at each position

    index = 0
    remaining = total

    for i in range(NUM_POINTS - 1):  # Last point is determined
        # How many positions have smaller value at point i?
        for v in range(pos[i]):
            # If we put v checkers on point i, remaining - v go on points i+1 to end
            # Number of ways = C(remaining - v + (NUM_POINTS - i - 1) - 1, (NUM_POINTS - i - 1) - 1)
            points_left = NUM_POINTS - i - 1
            checkers_left = remaining - v
            index += binomial(checkers_left + points_left - 1, points_left - 1)
        remaining -= pos[i]

    return index


def index_to_position(idx: int) -> Tuple[int, ...]:
    """Convert index back to position tuple.

    Args:
        idx: Index in [0, TOTAL_ONE_SIDED_POSITIONS).

    Returns:
        Tuple of 6 integers (checkers on points 1-6).
    """
    assert 0 <= idx < TOTAL_ONE_SIDED_POSITIONS

    # Find total checker count
    offset = 0
    total = 0
    for k in range(MAX_CHECKERS + 1):
        count = positions_for_checkers(k, NUM_POINTS)
        if offset + count > idx:
            total = k
            break
        offset += count

    inner_idx = idx - offset
    return _inner_index_to_pos(inner_idx, total)


def _inner_index_to_pos(idx: int, total: int) -> Tuple[int, ...]:
    """Convert inner index to position tuple."""
    pos = []
    remaining = total

    for i in range(NUM_POINTS - 1):
        # Find value at point i
        points_left = NUM_POINTS - i - 1
        cumulative = 0
        for v in range(remaining + 1):
            checkers_left = remaining - v
            count = binomial(checkers_left + points_left - 1, points_left - 1)
            if cumulative + count > idx:
                pos.append(v)
                idx -= cumulative
                remaining -= v
                break
            cumulative += count
        else:
            pos.append(remaining)
            remaining = 0

    # Last point gets remaining checkers
    pos.append(remaining)
    return tuple(pos)


# Precompute lookup tables for fast JAX-compatible indexing
# These allow vectorized position-to-index conversion

def _build_offset_table() -> np.ndarray:
    """Build table of offsets for each total checker count."""
    offsets = np.zeros(MAX_CHECKERS + 2, dtype=np.int64)
    for k in range(MAX_CHECKERS + 1):
        offsets[k + 1] = offsets[k] + positions_for_checkers(k, NUM_POINTS)
    return offsets

OFFSET_TABLE = _build_offset_table()


def _build_inner_index_coefficients() -> np.ndarray:
    """Build coefficient table for fast inner index computation.

    For each (point_idx, remaining_checkers, value_at_point),
    store the number of positions skipped.
    """
    # Shape: (NUM_POINTS, MAX_CHECKERS+1, MAX_CHECKERS+1)
    # coeff[i, r, v] = number of positions with value < v at point i,
    #                  given r remaining checkers for points i onwards
    coeff = np.zeros((NUM_POINTS, MAX_CHECKERS + 1, MAX_CHECKERS + 1), dtype=np.int64)

    for i in range(NUM_POINTS - 1):
        points_left = NUM_POINTS - i - 1
        for r in range(MAX_CHECKERS + 1):
            cumsum = 0
            for v in range(r + 1):
                coeff[i, r, v] = cumsum
                checkers_left = r - v
                cumsum += binomial(checkers_left + points_left - 1, points_left - 1)

    return coeff

INNER_INDEX_COEFFICIENTS = _build_inner_index_coefficients()


def position_to_index_fast(pos: np.ndarray) -> int:
    """Fast position to index using precomputed tables.

    Args:
        pos: Array of shape (6,) with checker counts.

    Returns:
        Index as integer.
    """
    total = int(np.sum(pos))
    offset = OFFSET_TABLE[total]

    inner_idx = 0
    remaining = total
    for i in range(NUM_POINTS - 1):
        v = int(pos[i])
        inner_idx += INNER_INDEX_COEFFICIENTS[i, remaining, v]
        remaining -= v

    return int(offset + inner_idx)


# JAX-compatible version using precomputed tables
_OFFSET_TABLE_JAX = jnp.array(OFFSET_TABLE)
_INNER_COEFF_JAX = jnp.array(INNER_INDEX_COEFFICIENTS)


@jax.jit
def position_to_index_jax(pos: jnp.ndarray) -> jnp.ndarray:
    """JAX-compatible position to index.

    Args:
        pos: Array of shape (6,) with checker counts.

    Returns:
        Index as jax array (scalar).
    """
    total = jnp.sum(pos).astype(jnp.int32)
    offset = _OFFSET_TABLE_JAX[total]

    # Compute inner index using cumulative remaining checkers
    def scan_fn(remaining, i_and_v):
        i, v = i_and_v
        contrib = _INNER_COEFF_JAX[i, remaining, v]
        new_remaining = remaining - v
        return new_remaining, contrib

    indices = jnp.arange(NUM_POINTS - 1)
    values = pos[:-1].astype(jnp.int32)
    _, contribs = jax.lax.scan(scan_fn, total, (indices, values))
    inner_idx = jnp.sum(contribs)

    return (offset + inner_idx).astype(jnp.int64)


# Verification
def verify_indexing():
    """Verify that indexing is bijective."""
    print(f"Total one-sided positions: {TOTAL_ONE_SIDED_POSITIONS:,}")

    # Test round-trip for some positions
    test_positions = [
        (0, 0, 0, 0, 0, 0),   # 0 checkers
        (1, 0, 0, 0, 0, 0),   # 1 checker on point 1
        (0, 0, 0, 0, 0, 1),   # 1 checker on point 6
        (15, 0, 0, 0, 0, 0),  # All on point 1
        (0, 0, 0, 0, 0, 15),  # All on point 6
        (3, 3, 3, 2, 2, 2),   # Spread
        (2, 2, 2, 3, 3, 3),   # Spread reverse
    ]

    print("\nRound-trip tests:")
    for pos in test_positions:
        idx = position_to_index(pos)
        recovered = index_to_position(idx)
        fast_idx = position_to_index_fast(np.array(pos))
        jax_idx = int(position_to_index_jax(jnp.array(pos)))
        status = "OK" if recovered == pos and fast_idx == idx and jax_idx == idx else "FAIL"
        print(f"  {pos} -> {idx} -> {recovered} (fast={fast_idx}, jax={jax_idx}) [{status}]")

    # Verify all indices are unique (for small subset)
    print("\nVerifying index uniqueness for 0-6 checkers...")
    seen = set()
    for total in range(7):
        for pos in _generate_positions(total, NUM_POINTS):
            idx = position_to_index(pos)
            if idx in seen:
                print(f"  DUPLICATE: {pos} -> {idx}")
                return False
            seen.add(idx)
    print(f"  Verified {len(seen)} unique indices")
    return True


def _generate_positions(total: int, points: int):
    """Generate all positions with given total checkers."""
    if points == 1:
        yield (total,)
        return
    for v in range(total + 1):
        for rest in _generate_positions(total - v, points - 1):
            yield (v,) + rest


if __name__ == "__main__":
    verify_indexing()
