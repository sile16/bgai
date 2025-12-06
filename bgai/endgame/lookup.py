"""Fast bearoff table lookup for training integration.

Provides JAX-compatible lookup into pre-computed bearoff tables.
Handles symmetry optimization for memory efficiency.
"""

import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from typing import Optional, Tuple
from pathlib import Path

from .indexing import (
    position_to_index_fast,
    position_to_index_jax,
    TOTAL_ONE_SIDED_POSITIONS,
    NUM_POINTS,
)


class BearoffTable:
    """Memory-efficient bearoff table with symmetry optimization.

    Stores only upper triangle (i <= j) since:
    V(X, O) for i > j can be computed as 1 - V(O, X)

    Supports both numpy and JAX lookups.
    """

    def __init__(self, table_path: Optional[str] = None, table: Optional[np.ndarray] = None):
        """Initialize from file or array.

        Args:
            table_path: Path to .npy file with table data.
            table: Direct array (either upper triangle or full matrix).
        """
        self.n = TOTAL_ONE_SIDED_POSITIONS

        if table_path is not None:
            self.table = np.load(table_path)
        elif table is not None:
            self.table = table
        else:
            raise ValueError("Must provide either table_path or table")

        # Detect format: upper triangle (1D) or full (2D)
        if self.table.ndim == 1:
            expected_size = self.n * (self.n + 1) // 2
            if self.table.shape[0] != expected_size:
                raise ValueError(f"Upper triangle size mismatch: {self.table.shape[0]} vs {expected_size}")
            self.format = 'upper_triangle'
        elif self.table.ndim == 2:
            if self.table.shape != (self.n, self.n):
                raise ValueError(f"Full table size mismatch: {self.table.shape} vs ({self.n}, {self.n})")
            self.format = 'full'
        else:
            raise ValueError(f"Unexpected table shape: {self.table.shape}")

        # Convert to JAX array for fast lookup
        self._table_jax = jnp.array(self.table)

        print(f"Loaded bearoff table: format={self.format}, shape={self.table.shape}")
        print(f"Memory: {self.table.nbytes / 1e6:.1f} MB")

    def lookup(self, x_pos: np.ndarray, o_pos: np.ndarray) -> float:
        """Look up win probability for X given positions.

        Args:
            x_pos: X's position (6-element array)
            o_pos: O's position (6-element array)

        Returns:
            P(X wins | X to move)
        """
        x_idx = position_to_index_fast(x_pos)
        o_idx = position_to_index_fast(o_pos)

        if self.format == 'full':
            return float(self.table[x_idx, o_idx])
        else:
            return self._lookup_upper_triangle(x_idx, o_idx)

    def _lookup_upper_triangle(self, i: int, j: int) -> float:
        """Lookup in upper triangle format with symmetry."""
        if i <= j:
            idx = i * self.n - i * (i + 1) // 2 + j
            return float(self.table[idx])
        else:
            # Use symmetry: V(i, j) = 1 - V(j, i)
            idx = j * self.n - j * (j + 1) // 2 + i
            return 1.0 - float(self.table[idx])

    @partial(jax.jit, static_argnums=0)
    def lookup_jax(self, x_pos: jnp.ndarray, o_pos: jnp.ndarray) -> jnp.ndarray:
        """JAX-compatible lookup (JIT-compiled).

        Args:
            x_pos: X's position (6-element array)
            o_pos: O's position (6-element array)

        Returns:
            P(X wins | X to move) as JAX scalar
        """
        x_idx = position_to_index_jax(x_pos)
        o_idx = position_to_index_jax(o_pos)

        if self.format == 'full':
            return self._table_jax[x_idx, o_idx]
        else:
            return self._lookup_upper_triangle_jax(x_idx, o_idx)

    def _lookup_upper_triangle_jax(self, i: jnp.ndarray, j: jnp.ndarray) -> jnp.ndarray:
        """JAX lookup in upper triangle with symmetry."""
        # Compute both possible indices
        idx_ij = i * self.n - i * (i + 1) // 2 + j
        idx_ji = j * self.n - j * (j + 1) // 2 + i

        # Use lax.cond for branching
        return jax.lax.cond(
            i <= j,
            lambda _: self._table_jax[idx_ij],
            lambda _: 1.0 - self._table_jax[idx_ji],
            operand=None
        )

    def is_bearoff_position(self, board: np.ndarray) -> bool:
        """Check if PGX board state is a bearoff position.

        Bearoff requires:
        - All X checkers on points 0-5 (X's home board) or borne off
        - All O checkers on points 18-23 (O's home board) or borne off
        - No checkers on bar

        Args:
            board: PGX board array (28 elements)

        Returns:
            True if position is in bearoff database
        """
        # Check bar is empty
        if board[24] != 0 or board[25] != 0:
            return False

        # Check X has no checkers outside home board
        for i in range(6, 24):
            if board[i] > 0:
                return False

        # Check O has no checkers outside their home board
        for i in range(0, 18):
            if board[i] < 0:
                return False

        return True

    def get_perfect_value_from_pgx(self, board: np.ndarray) -> Optional[float]:
        """Get perfect value from PGX board if in bearoff.

        Args:
            board: PGX board array (28 elements)

        Returns:
            Perfect win probability, or None if not bearoff.
        """
        if not self.is_bearoff_position(board):
            return None

        # Extract X's home board position (points 0-5)
        x_pos = np.array([max(0, board[i]) for i in range(6)])

        # Extract O's home board position (points 18-23, stored as negative)
        # O's point 1 = pgx point 23, O's point 6 = pgx point 18
        o_pos = np.array([max(0, -board[23-i]) for i in range(6)])

        return self.lookup(x_pos, o_pos)


def convert_full_to_upper_triangle(full_table: np.ndarray) -> np.ndarray:
    """Convert full (n,n) table to upper triangle (1D) format."""
    n = full_table.shape[0]
    size = n * (n + 1) // 2
    upper = np.zeros(size, dtype=full_table.dtype)

    idx = 0
    for i in range(n):
        for j in range(i, n):
            upper[idx] = full_table[i, j]
            idx += 1

    return upper


def convert_to_uint16(table: np.ndarray) -> np.ndarray:
    """Convert float table to uint16 for storage efficiency.

    Maps [0.0, 1.0] -> [0, 65535]
    """
    return (table * 65535).astype(np.uint16)


def convert_from_uint16(table: np.ndarray) -> np.ndarray:
    """Convert uint16 table back to float32."""
    return table.astype(np.float32) / 65535.0


class BearoffTableOptimized:
    """Memory-efficient bearoff table with upper triangle storage and vectorized lookups.

    Stores only upper triangle (i <= j) using the formula:
        idx = i * n - i*(i+1)/2 + j  for i <= j

    For i > j, we use symmetry: V(i, j) = 1 - V(j, i)

    This reduces storage from n*n to n*(n+1)/2, nearly halving memory usage.
    """

    def __init__(
        self,
        table_path: Optional[str] = None,
        full_table: Optional[np.ndarray] = None,
        upper_table: Optional[np.ndarray] = None,
    ):
        """Initialize from file or array.

        Args:
            table_path: Path to .npy file (either full or upper triangle format).
            full_table: Full (n, n) table - will be converted to upper triangle.
            upper_table: Pre-converted upper triangle (1D array).
        """
        self.n = TOTAL_ONE_SIDED_POSITIONS

        if table_path is not None:
            raw = np.load(table_path)
            if raw.ndim == 2:
                # Full table - convert to upper triangle
                print(f"Converting full table to upper triangle format...")
                self.table = self._extract_upper_triangle(raw)
            else:
                self.table = raw
        elif full_table is not None:
            self.table = self._extract_upper_triangle(full_table)
        elif upper_table is not None:
            self.table = upper_table
        else:
            raise ValueError("Must provide table_path, full_table, or upper_table")

        # Verify size
        expected_size = self.n * (self.n + 1) // 2
        if self.table.shape[0] != expected_size:
            raise ValueError(f"Upper triangle size mismatch: {self.table.shape[0]} vs {expected_size}")

        # Ensure float32 for fast lookup
        if self.table.dtype != np.float32:
            self.table = self.table.astype(np.float32)

        print(f"BearoffTableOptimized: n={self.n:,}, size={len(self.table):,}, "
              f"memory={self.table.nbytes / 1e9:.2f} GB")

    def _extract_upper_triangle(self, full_table: np.ndarray) -> np.ndarray:
        """Extract upper triangle to 1D array."""
        n = full_table.shape[0]
        size = n * (n + 1) // 2
        upper = np.zeros(size, dtype=np.float32)

        idx = 0
        for i in range(n):
            # Copy row i from column i to n
            row_len = n - i
            upper[idx:idx + row_len] = full_table[i, i:]
            idx += row_len

        return upper

    def _compute_index(self, i: int, j: int) -> int:
        """Compute 1D index for position (i, j) where i <= j."""
        # idx = i * n - i*(i+1)/2 + j
        return i * self.n - (i * (i + 1)) // 2 + j

    def lookup(self, x_idx: int, o_idx: int) -> float:
        """Look up win probability for X given position indices.

        Args:
            x_idx: X's position index
            o_idx: O's position index

        Returns:
            P(X wins | X to move)
        """
        if x_idx <= o_idx:
            idx = self._compute_index(x_idx, o_idx)
            return float(self.table[idx])
        else:
            # Symmetry: V(i, j) = 1 - V(j, i)
            idx = self._compute_index(o_idx, x_idx)
            return 1.0 - float(self.table[idx])

    def lookup_batch(
        self,
        x_indices: np.ndarray,
        o_indices: np.ndarray
    ) -> np.ndarray:
        """Vectorized lookup for batched position indices.

        Args:
            x_indices: Array of X position indices (batch_size,)
            o_indices: Array of O position indices (batch_size,)

        Returns:
            Array of win probabilities (batch_size,)
        """
        x_indices = x_indices.astype(np.int64)
        o_indices = o_indices.astype(np.int64)

        # Determine which indices need to be swapped (symmetry)
        swap_mask = x_indices > o_indices

        # Compute i, j where i <= j
        i = np.where(swap_mask, o_indices, x_indices)
        j = np.where(swap_mask, x_indices, o_indices)

        # Compute 1D indices: idx = i * n - i*(i+1)/2 + j
        flat_indices = i * self.n - (i * (i + 1)) // 2 + j

        # Lookup values
        values = self.table[flat_indices]

        # Apply symmetry correction where swapped
        values = np.where(swap_mask, 1.0 - values, values)

        return values

    def save(self, path: str):
        """Save upper triangle table to file."""
        np.save(path, self.table)
        print(f"Saved upper triangle table to {path}")

    @classmethod
    def from_full_table_file(cls, full_path: str, save_path: Optional[str] = None):
        """Create from full table file, optionally saving upper triangle version.

        Args:
            full_path: Path to full (n, n) table file
            save_path: Optional path to save upper triangle version

        Returns:
            BearoffTableOptimized instance
        """
        print(f"Loading full table from {full_path}...")
        full_table = np.load(full_path)
        instance = cls(full_table=full_table)

        if save_path:
            instance.save(save_path)

        return instance


def create_optimized_table(full_table_path: str, output_path: Optional[str] = None) -> str:
    """Convert full bearoff table to optimized upper triangle format.

    Args:
        full_table_path: Path to full (n, n) .npy file
        output_path: Output path (default: same location with _opt suffix)

    Returns:
        Path to saved optimized table
    """
    if output_path is None:
        base = Path(full_table_path)
        output_path = str(base.parent / f"{base.stem}_opt.npy")

    table = BearoffTableOptimized.from_full_table_file(full_table_path, output_path)
    return output_path
