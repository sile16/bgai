"""Endgame tables for perfect bearoff play.

This module provides:
- Native JAX-compatible position indexing
- Pre-computed two-sided bearoff tables
- Memory-efficient storage with symmetry optimization

Target: ~3 billion positions in 1-3 GB using:
- Symmetry: P(X wins | X, O) = 1 - P(X wins | O, X)
- uint16 storage: 0-65535 maps to 0.0-1.0
- Combinatorial perfect hashing for O(1) indexing
"""

from .indexing import (
    position_to_index,
    index_to_position,
    total_positions,
    TOTAL_ONE_SIDED_POSITIONS,
)
from .lookup import BearoffTable

__all__ = [
    'position_to_index',
    'index_to_position',
    'total_positions',
    'TOTAL_ONE_SIDED_POSITIONS',
    'BearoffTable',
]
