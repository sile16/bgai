"""Utilities for packed upper-triangle bearoff tables."""

import math
from typing import Tuple

import numpy as np


def upper_triangle_size(n: int) -> int:
    """Number of entries in the upper triangle (inclusive)."""
    return n * (n + 1) // 2


def solve_size_to_n(entries: int) -> int:
    """Infer n from number of packed entries."""
    n = int((math.isqrt(1 + 8 * entries) - 1) // 2)
    if upper_triangle_size(n) != entries:
        raise ValueError(f"Packed length {entries} does not correspond to a valid n")
    return n


def upper_index(i: int, j: int, n: int) -> int:
    """Return flat index for (i, j) with i <= j in packed upper triangle."""
    # Rows are stored consecutively: row0 has n entries, row1 has n-1, ...
    # Start of row i = sum_{k=0}^{i-1} (n-k) = i*n - i*(i-1)/2
    return i * n - (i * (i - 1)) // 2 + (j - i)


def mirror_probs(probs: np.ndarray) -> np.ndarray:
    """Swap perspective for cubeless probs and optional cube equities.

    Layout supported:
        4: [win, gw, loss, gl]
        7: [win, gw, loss, gl, eq_center, eq_owner, eq_opponent]
    """
    if probs.shape[-1] == 4:
        win, gw, loss, gl = probs
        return np.array([loss, gl, win, gw], dtype=probs.dtype)

    if probs.shape[-1] == 7:
        win, gw, loss, gl, eq_c, eq_own, eq_opp = probs
        # Equities are zero-sum: flip sign, swap owner/opponent
        return np.array(
            [loss, gl, win, gw, -eq_c, -eq_opp, -eq_own],
            dtype=probs.dtype,
        )

    raise ValueError(f"Unsupported probability vector length {probs.shape[-1]}")


class BearoffLookup:
    """Lookup wrapper for packed or full bearoff tables.

    Tables store probabilities (len 4 or 7) from the perspective of the first
    index (X) with X to move.
    """

    def __init__(self, table: np.ndarray, packed: bool, n: int):
        self.table = table
        self.packed = packed
        self.n = n
        self.last_dim = table.shape[-1]

    def probs_for(self, x_idx: int, o_idx: int) -> np.ndarray:
        """Return probs for X to move given indices."""
        if not self.packed:
            return np.array(self.table[x_idx, o_idx], dtype=np.float32)

        if x_idx <= o_idx:
            idx = upper_index(x_idx, o_idx, self.n)
            return np.array(self.table[idx], dtype=np.float32)

        idx = upper_index(o_idx, x_idx, self.n)
        return mirror_probs(self.table[idx])

    def probs_for_player(self, x_idx: int, o_idx: int, cur_player: int) -> np.ndarray:
        """Return probs from the perspective of cur_player (0=X,1=O)."""
        if cur_player == 0:
            return self.probs_for(x_idx, o_idx)
        return self.probs_for(o_idx, x_idx)

    def cube_equities(self, x_idx: int, o_idx: int, cur_player: int) -> Tuple[float, float, float]:
        """Return (centered, owner, opponent) cube equities if present, else zeros."""
        probs = self.probs_for_player(x_idx, o_idx, cur_player)
        if probs.shape[-1] < 7:
            return 0.0, 0.0, 0.0
        _, _, _, _, eq_c, eq_owner, eq_opp = probs
        return float(eq_c), float(eq_owner), float(eq_opp)


def pack_upper(table: np.ndarray) -> np.ndarray:
    """Pack a full (n,n,last_dim) table into upper-triangle (entries,last_dim)."""
    assert table.ndim == 3 and table.shape[0] == table.shape[1]
    n = table.shape[0]
    packed = np.zeros((upper_triangle_size(n), table.shape[2]), dtype=table.dtype)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            packed[idx] = table[i, j]
            idx += 1
    return packed
