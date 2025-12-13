"""Utilities for packed upper-triangle bearoff tables.

Version 2 supports tiered dual-perspective storage with configurable precision loading.
"""

import json
import math
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, Union

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
        return np.array(
            [loss, gl, win, gw, -eq_c, -eq_opp, -eq_own],
            dtype=probs.dtype,
        )

    raise ValueError(f"Unsupported probability vector length {probs.shape[-1]}")


class BearoffLookup:
    """Lookup wrapper for packed or full bearoff tables (v1 format).

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


# ========== Version 2: Tiered dual-perspective format ==========


class Precision(Enum):
    """Memory precision modes for loading bearoff tables."""
    UINT24 = "uint24"  # Full precision: 3 bytes per value (uint8 triplets)
    UINT16 = "uint16"  # Reduced: 2 bytes per value
    FLOAT32 = "float32"  # Decoded floats: 4 bytes per value


class TieredHeader(NamedTuple):
    """Header for tiered dual-perspective bearoff tables (v2)."""
    version: int
    max_checkers: int
    max_points: int
    n_positions: int
    tier1_entries: int
    tier1_values_per_entry: int
    tier2_entries: int
    tier2_values_per_entry: int


def decode_uint24_to_float(raw: np.ndarray, is_equity: bool = False) -> np.ndarray:
    """Decode uint24 (stored as 3 uint8s) to float32.

    Args:
        raw: Array of shape (..., 3) with uint8 values (little endian uint24)
        is_equity: If True, decode to [-1, 1], else decode to [0, 1]
    """
    # Combine 3 bytes into uint32
    vals = raw[..., 0].astype(np.uint32)
    vals |= raw[..., 1].astype(np.uint32) << 8
    vals |= raw[..., 2].astype(np.uint32) << 16

    # Normalize to float
    floats = vals.astype(np.float32) / 16777215.0

    if is_equity:
        return floats * 2.0 - 1.0
    return floats


def decode_uint24_to_uint16(raw: np.ndarray) -> np.ndarray:
    """Truncate uint24 (stored as 3 uint8s) to uint16.

    Takes the upper 16 bits of the 24-bit value.
    """
    # Combine into uint24, then take upper 16 bits
    vals = raw[..., 0].astype(np.uint32)
    vals |= raw[..., 1].astype(np.uint32) << 8
    vals |= raw[..., 2].astype(np.uint32) << 16
    return (vals >> 8).astype(np.uint16)


class TieredBearoffLookup:
    """Lookup for tiered dual-perspective bearoff tables (v2 format).

    Version 2 stores BOTH perspectives for each upper-triangle entry:
    - Tier 1 (no gammons): positions where both players have borne off ≥1 checker
    - Tier 2 (gammons possible): at least one player has all checkers on board

    Layout per entry:
    - Tier 1: [win_ij, eq_cl_ij, eq_own_ij, eq_ctr_ij, eq_opp_ij,
               win_ji, eq_cl_ji, eq_own_ji, eq_ctr_ji, eq_opp_ji]
    - Tier 2: [win_ij, gam_win_ij, gam_loss_ij, eq_cl_ij, eq_own_ij, eq_ctr_ij, eq_opp_ij,
               win_ji, gam_win_ji, gam_loss_ji, eq_cl_ji, eq_own_ji, eq_ctr_ji, eq_opp_ji]

    Precision modes control memory vs accuracy tradeoff:
    - UINT24: Keep raw bytes, decode on lookup (lowest memory for storage)
    - UINT16: Downsample to uint16, decode on lookup (moderate memory)
    - FLOAT32: Pre-decode to float32 (fastest lookup, highest memory)
    """

    TIER1_VALUES = 10  # 5 values per perspective
    TIER2_VALUES = 14  # 7 values per perspective

    def __init__(
        self,
        header: TieredHeader,
        tier1_raw: np.ndarray,
        tier2_raw: np.ndarray,
        position_sums: np.ndarray,
        precision: Precision = Precision.FLOAT32,
    ):
        self.header = header
        self.n = header.n_positions
        self.max_checkers = header.max_checkers
        self.position_sums = position_sums
        self.precision = precision

        # Build tier1/tier2 index mapping
        # For each upper-triangle entry (i,j) with i<=j, we need to know if it's tier1 or tier2
        # and what its index within that tier is
        self._build_tier_indices()

        # Process arrays based on precision
        if precision == Precision.FLOAT32:
            self.tier1 = self._decode_tier1_to_float(tier1_raw)
            self.tier2 = self._decode_tier2_to_float(tier2_raw)
        elif precision == Precision.UINT16:
            self.tier1 = self._decode_tier_to_uint16(tier1_raw, self.TIER1_VALUES)
            self.tier2 = self._decode_tier_to_uint16(tier2_raw, self.TIER2_VALUES)
        else:  # UINT24 - keep raw
            self.tier1 = tier1_raw
            self.tier2 = tier2_raw

    def _build_tier_indices(self):
        """Build mapping from (i,j) to tier and index within tier."""
        n = self.n
        n_upper = upper_triangle_size(n)

        # For each upper-triangle position, store (is_tier2, index_in_tier)
        self._tier_map = np.zeros((n_upper, 2), dtype=np.int32)

        tier1_idx = 0
        tier2_idx = 0

        for i in range(n):
            for j in range(i, n):
                flat_idx = upper_index(i, j, n)
                gammon_possible = (
                    self.position_sums[i] == self.max_checkers or
                    self.position_sums[j] == self.max_checkers
                )

                if gammon_possible:
                    self._tier_map[flat_idx] = [1, tier2_idx]
                    tier2_idx += 1
                else:
                    self._tier_map[flat_idx] = [0, tier1_idx]
                    tier1_idx += 1

    def _decode_tier1_to_float(self, raw: np.ndarray) -> np.ndarray:
        """Decode tier1 raw uint24 array to float32.

        Tier 1 layout: [win_ij, eq_cl_ij, eq_own_ij, eq_ctr_ij, eq_opp_ij,
                        win_ji, eq_cl_ji, eq_own_ji, eq_ctr_ji, eq_opp_ji]
        """
        n_entries = raw.shape[0]
        result = np.zeros((n_entries, self.TIER1_VALUES), dtype=np.float32)

        # Reshape to (entries, values, 3)
        raw_3d = raw.reshape(n_entries, self.TIER1_VALUES, 3)

        # Win probs: indices 0, 5
        result[:, 0] = decode_uint24_to_float(raw_3d[:, 0], is_equity=False)
        result[:, 5] = decode_uint24_to_float(raw_3d[:, 5], is_equity=False)

        # Equities: indices 1-4, 6-9
        for idx in [1, 2, 3, 4, 6, 7, 8, 9]:
            result[:, idx] = decode_uint24_to_float(raw_3d[:, idx], is_equity=True)

        return result

    def _decode_tier2_to_float(self, raw: np.ndarray) -> np.ndarray:
        """Decode tier2 raw uint24 array to float32.

        Tier 2 layout: [win_ij, gam_win_ij, gam_loss_ij, eq_cl_ij, eq_own_ij, eq_ctr_ij, eq_opp_ij,
                        win_ji, gam_win_ji, gam_loss_ji, eq_cl_ji, eq_own_ji, eq_ctr_ji, eq_opp_ji]
        """
        n_entries = raw.shape[0]
        result = np.zeros((n_entries, self.TIER2_VALUES), dtype=np.float32)

        raw_3d = raw.reshape(n_entries, self.TIER2_VALUES, 3)

        # Probabilities: indices 0,1,2 (ij) and 7,8,9 (ji)
        for idx in [0, 1, 2, 7, 8, 9]:
            result[:, idx] = decode_uint24_to_float(raw_3d[:, idx], is_equity=False)

        # Equities: indices 3,4,5,6 (ij) and 10,11,12,13 (ji)
        for idx in [3, 4, 5, 6, 10, 11, 12, 13]:
            result[:, idx] = decode_uint24_to_float(raw_3d[:, idx], is_equity=True)

        return result

    def _decode_tier_to_uint16(self, raw: np.ndarray, n_values: int) -> np.ndarray:
        """Downsample uint24 to uint16."""
        n_entries = raw.shape[0]
        raw_3d = raw.reshape(n_entries, n_values, 3)
        return decode_uint24_to_uint16(raw_3d)

    def _decode_single_uint16(self, val: np.uint16, is_equity: bool) -> float:
        """Decode single uint16 to float."""
        f = float(val) / 65535.0
        return f * 2.0 - 1.0 if is_equity else f

    def _decode_single_uint24(self, bytes3: np.ndarray, is_equity: bool) -> float:
        """Decode single uint24 (3 bytes) to float."""
        val = int(bytes3[0]) | (int(bytes3[1]) << 8) | (int(bytes3[2]) << 16)
        f = val / 16777215.0
        return f * 2.0 - 1.0 if is_equity else f

    def _get_tier1_values(self, idx: int, perspective: int) -> Tuple[float, float, float, float, float]:
        """Get (win, eq_cl, eq_own, eq_ctr, eq_opp) from tier1 entry.

        perspective: 0 = ij, 1 = ji
        """
        offset = 5 * perspective

        if self.precision == Precision.FLOAT32:
            row = self.tier1[idx]
            return (row[offset], row[offset + 1], row[offset + 2],
                    row[offset + 3], row[offset + 4])

        elif self.precision == Precision.UINT16:
            row = self.tier1[idx]
            win = self._decode_single_uint16(row[offset], False)
            eq_cl = self._decode_single_uint16(row[offset + 1], True)
            eq_own = self._decode_single_uint16(row[offset + 2], True)
            eq_ctr = self._decode_single_uint16(row[offset + 3], True)
            eq_opp = self._decode_single_uint16(row[offset + 4], True)
            return win, eq_cl, eq_own, eq_ctr, eq_opp

        else:  # UINT24
            raw = self.tier1[idx].reshape(self.TIER1_VALUES, 3)
            win = self._decode_single_uint24(raw[offset], False)
            eq_cl = self._decode_single_uint24(raw[offset + 1], True)
            eq_own = self._decode_single_uint24(raw[offset + 2], True)
            eq_ctr = self._decode_single_uint24(raw[offset + 3], True)
            eq_opp = self._decode_single_uint24(raw[offset + 4], True)
            return win, eq_cl, eq_own, eq_ctr, eq_opp

    def _get_tier2_values(self, idx: int, perspective: int) -> Tuple[float, float, float, float, float, float, float]:
        """Get (win, gam_win, gam_loss, eq_cl, eq_own, eq_ctr, eq_opp) from tier2 entry.

        perspective: 0 = ij, 1 = ji
        """
        offset = 7 * perspective

        if self.precision == Precision.FLOAT32:
            row = self.tier2[idx]
            return (row[offset], row[offset + 1], row[offset + 2],
                    row[offset + 3], row[offset + 4], row[offset + 5], row[offset + 6])

        elif self.precision == Precision.UINT16:
            row = self.tier2[idx]
            win = self._decode_single_uint16(row[offset], False)
            gam_win = self._decode_single_uint16(row[offset + 1], False)
            gam_loss = self._decode_single_uint16(row[offset + 2], False)
            eq_cl = self._decode_single_uint16(row[offset + 3], True)
            eq_own = self._decode_single_uint16(row[offset + 4], True)
            eq_ctr = self._decode_single_uint16(row[offset + 5], True)
            eq_opp = self._decode_single_uint16(row[offset + 6], True)
            return win, gam_win, gam_loss, eq_cl, eq_own, eq_ctr, eq_opp

        else:  # UINT24
            raw = self.tier2[idx].reshape(self.TIER2_VALUES, 3)
            win = self._decode_single_uint24(raw[offset], False)
            gam_win = self._decode_single_uint24(raw[offset + 1], False)
            gam_loss = self._decode_single_uint24(raw[offset + 2], False)
            eq_cl = self._decode_single_uint24(raw[offset + 3], True)
            eq_own = self._decode_single_uint24(raw[offset + 4], True)
            eq_ctr = self._decode_single_uint24(raw[offset + 5], True)
            eq_opp = self._decode_single_uint24(raw[offset + 6], True)
            return win, gam_win, gam_loss, eq_cl, eq_own, eq_ctr, eq_opp

    def lookup(self, i: int, j: int) -> Tuple[float, float, float, float, float, float, float]:
        """Look up values for position (i, j) where i is player-to-move's position.

        Returns: (win, gammon_win, gammon_loss, eq_cubeless, eq_owner, eq_center, eq_opponent)

        Note: gammon_win and gammon_loss are 0.0 for tier1 positions.
        """
        # Determine upper triangle index and perspective
        if i <= j:
            flat_idx = upper_index(i, j, self.n)
            perspective = 0  # ij perspective
        else:
            flat_idx = upper_index(j, i, self.n)
            perspective = 1  # ji perspective

        is_tier2, tier_idx = self._tier_map[flat_idx]

        if is_tier2:
            return self._get_tier2_values(tier_idx, perspective)
        else:
            win, eq_cl, eq_own, eq_ctr, eq_opp = self._get_tier1_values(tier_idx, perspective)
            return win, 0.0, 0.0, eq_cl, eq_own, eq_ctr, eq_opp

    def probs_for(self, x_idx: int, o_idx: int) -> np.ndarray:
        """Return probs array for compatibility with BearoffLookup.

        Returns 7-element array: [win, gam_win, loss, gam_loss, eq_center, eq_owner, eq_opponent]
        """
        win, gam_win, gam_loss, eq_cl, eq_own, eq_ctr, eq_opp = self.lookup(x_idx, o_idx)
        loss = 1.0 - win
        return np.array([win, gam_win, loss, gam_loss, eq_ctr, eq_own, eq_opp], dtype=np.float32)

    def probs_for_player(self, x_idx: int, o_idx: int, cur_player: int) -> np.ndarray:
        """Return probs from the perspective of cur_player (0=X, 1=O)."""
        if cur_player == 0:
            return self.probs_for(x_idx, o_idx)
        return self.probs_for(o_idx, x_idx)

    def cube_equities(self, x_idx: int, o_idx: int, cur_player: int) -> Tuple[float, float, float]:
        """Return (centered, owner, opponent) cube equities."""
        if cur_player == 0:
            _, _, _, _, eq_own, eq_ctr, eq_opp = self.lookup(x_idx, o_idx)
        else:
            _, _, _, _, eq_own, eq_ctr, eq_opp = self.lookup(o_idx, x_idx)
        return eq_ctr, eq_own, eq_opp

    def memory_usage_bytes(self) -> int:
        """Return approximate memory usage in bytes."""
        if self.precision == Precision.FLOAT32:
            return (self.tier1.nbytes + self.tier2.nbytes +
                    self._tier_map.nbytes + self.position_sums.nbytes)
        elif self.precision == Precision.UINT16:
            return (self.tier1.nbytes + self.tier2.nbytes +
                    self._tier_map.nbytes + self.position_sums.nbytes)
        else:
            return (self.tier1.nbytes + self.tier2.nbytes +
                    self._tier_map.nbytes + self.position_sums.nbytes)


def compute_position_sums(n_positions: int, max_checkers: int) -> np.ndarray:
    """Compute position sums for gnubg bearoff ordering."""
    from scipy.special import comb

    def position_inv(id_: int, n: int, r: int) -> int:
        if r == 0:
            return 0
        if n == r:
            return (1 << n) - 1
        nc = int(comb(n - 1, r, exact=True))
        if id_ >= nc:
            return (1 << (n - 1)) | position_inv(id_ - nc, n - 1, r - 1)
        return position_inv(id_, n - 1, r)

    NUM_POINTS = 6
    sums = np.zeros(n_positions, dtype=np.int32)

    for id_ in range(n_positions):
        fbits = position_inv(id_, max_checkers + NUM_POINTS, NUM_POINTS)
        pos = [0] * NUM_POINTS
        j = NUM_POINTS - 1
        for i in range(max_checkers + NUM_POINTS):
            if (fbits >> i) & 1:
                if j == 0:
                    break
                j -= 1
            else:
                pos[j] += 1
        sums[id_] = sum(pos)

    return sums


def load_tiered_bearoff(
    tier1_path: Union[str, Path],
    tier2_path: Union[str, Path],
    header_path: Union[str, Path],
    precision: Precision = Precision.FLOAT32,
) -> TieredBearoffLookup:
    """Load tiered bearoff tables from disk.

    Args:
        tier1_path: Path to tier1 .npy file
        tier2_path: Path to tier2 .npy file
        header_path: Path to header .json file
        precision: Memory precision mode
    """
    with open(header_path) as f:
        hdr = json.load(f)

    header = TieredHeader(
        version=hdr["version"],
        max_checkers=hdr["max_checkers"],
        max_points=hdr["max_points"],
        n_positions=hdr["n_positions"],
        tier1_entries=hdr["tier1_entries"],
        tier1_values_per_entry=hdr["tier1_values_per_entry"],
        tier2_entries=hdr["tier2_entries"],
        tier2_values_per_entry=hdr["tier2_values_per_entry"],
    )

    tier1_raw = np.load(tier1_path)
    tier2_raw = np.load(tier2_path)

    position_sums = compute_position_sums(header.n_positions, header.max_checkers)

    return TieredBearoffLookup(header, tier1_raw, tier2_raw, position_sums, precision)


def generate_and_save_tiered(
    max_checkers: int,
    output_dir: Union[str, Path],
    prefix: str = "bearoff_tiered",
) -> Tuple[str, str, str]:
    """Generate tiered bearoff tables and save to disk.

    Returns (tier1_path, tier2_path, header_path).
    """
    import bearoff

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    header_json, tier1_arr, tier2_arr = bearoff.generate_tiered_bearoff(max_checkers)

    tier1_path = output_dir / f"{prefix}_{max_checkers}_tier1.npy"
    tier2_path = output_dir / f"{prefix}_{max_checkers}_tier2.npy"
    header_path = output_dir / f"{prefix}_{max_checkers}.json"

    np.save(tier1_path, tier1_arr)
    np.save(tier2_path, tier2_arr)

    with open(header_path, "w") as f:
        f.write(header_json)

    return str(tier1_path), str(tier2_path), str(header_path)


# ========== Version 4: Full perspective with conditional gammons ==========


class V4Header(NamedTuple):
    """Header for V4 bearoff tables with conditional gammon probabilities."""
    version: int
    max_checkers: int
    max_points: int
    n_positions: int
    total_entries: int
    values_per_entry: int
    bytes_per_entry: int


class V4BearoffLookup:
    """Lookup for V4 bearoff tables with conditional gammon probabilities.

    V4 format stores 12 uint16 values per entry (24 bytes):
    - gam_win_cond_ij, gam_loss_cond_ij, eq_cl_ij, eq_own_ij, eq_ctr_ij, eq_opp_ij
    - gam_win_cond_ji, gam_loss_cond_ji, eq_cl_ji, eq_own_ji, eq_ctr_ji, eq_opp_ji

    Win probability is derived from cubeless equity: win = (eq_cl + 1) / 2
    Gammon probabilities are conditional: P(gammon|win) and P(gammon|loss)

    Both perspectives (i,j) and (j,i) are stored together in upper triangle format.
    """

    VALUES_PER_ENTRY = 12
    BYTES_PER_ENTRY = 24

    # Value indices for perspective 0 (ij)
    GAM_WIN_COND_IJ = 0
    GAM_LOSS_COND_IJ = 1
    EQ_CL_IJ = 2
    EQ_OWN_IJ = 3
    EQ_CTR_IJ = 4
    EQ_OPP_IJ = 5

    # Value indices for perspective 1 (ji)
    GAM_WIN_COND_JI = 6
    GAM_LOSS_COND_JI = 7
    EQ_CL_JI = 8
    EQ_OWN_JI = 9
    EQ_CTR_JI = 10
    EQ_OPP_JI = 11

    def __init__(
        self,
        header: V4Header,
        data: np.ndarray,
    ):
        """Initialize V4 lookup.

        Args:
            header: V4Header with table metadata
            data: Raw uint16 data of shape (total_entries, 12)
        """
        self.header = header
        self.n = header.n_positions
        self.data = data

    @classmethod
    def load(cls, data_path: Union[str, Path], header_path: Union[str, Path]) -> "V4BearoffLookup":
        """Load V4 bearoff table from disk.

        Args:
            data_path: Path to binary data file (.bin)
            header_path: Path to JSON header file
        """
        with open(header_path) as f:
            hdr = json.load(f)

        header = V4Header(
            version=hdr["version"],
            max_checkers=hdr["max_checkers"],
            max_points=hdr["max_points"],
            n_positions=hdr["n_positions"],
            total_entries=hdr["total_entries"],
            values_per_entry=hdr["values_per_entry"],
            bytes_per_entry=hdr["bytes_per_entry"],
        )

        data_path = Path(data_path)
        expected = header.total_entries * cls.VALUES_PER_ENTRY
        expected_bytes = expected * np.dtype(np.uint16).itemsize

        actual_bytes = data_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Bearoff V4 data size mismatch: expected {expected_bytes} bytes, got {actual_bytes}"
            )

        # Load binary data into RAM (fast lookups, higher memory).
        with open(data_path, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint16)

        data = raw.reshape(header.total_entries, cls.VALUES_PER_ENTRY)
        return cls(header, data)

    def _decode_prob(self, val: np.uint16) -> float:
        """Decode uint16 to probability [0, 1]."""
        return float(val) / 65535.0

    def _decode_equity(self, val: np.uint16) -> float:
        """Decode uint16 to equity [-1, 1]."""
        return (float(val) / 65535.0) * 2.0 - 1.0

    def lookup_raw(self, i: int, j: int) -> Tuple[np.ndarray, int]:
        """Get raw uint16 values and perspective for position (i, j).

        Args:
            i: Index of player to move
            j: Index of opponent

        Returns:
            (raw_values, perspective): raw uint16 array and perspective (0=ij, 1=ji)
        """
        if i <= j:
            flat_idx = upper_index(i, j, self.n)
            perspective = 0
        else:
            flat_idx = upper_index(j, i, self.n)
            perspective = 1

        return self.data[flat_idx], perspective

    def lookup(self, i: int, j: int) -> Tuple[float, float, float, float, float, float, float]:
        """Look up values for position (i, j) where i is player-to-move's position.

        Returns:
            (win, gam_win_cond, gam_loss_cond, eq_cl, eq_own, eq_ctr, eq_opp)
        """
        raw, perspective = self.lookup_raw(i, j)

        if perspective == 0:
            # Use ij values
            gam_win_cond = self._decode_prob(raw[self.GAM_WIN_COND_IJ])
            gam_loss_cond = self._decode_prob(raw[self.GAM_LOSS_COND_IJ])
            eq_cl = self._decode_equity(raw[self.EQ_CL_IJ])
            eq_own = self._decode_equity(raw[self.EQ_OWN_IJ])
            eq_ctr = self._decode_equity(raw[self.EQ_CTR_IJ])
            eq_opp = self._decode_equity(raw[self.EQ_OPP_IJ])
        else:
            # Use ji values
            gam_win_cond = self._decode_prob(raw[self.GAM_WIN_COND_JI])
            gam_loss_cond = self._decode_prob(raw[self.GAM_LOSS_COND_JI])
            eq_cl = self._decode_equity(raw[self.EQ_CL_JI])
            eq_own = self._decode_equity(raw[self.EQ_OWN_JI])
            eq_ctr = self._decode_equity(raw[self.EQ_CTR_JI])
            eq_opp = self._decode_equity(raw[self.EQ_OPP_JI])

        # Derive win from cubeless equity
        win = (eq_cl + 1.0) / 2.0

        return win, gam_win_cond, gam_loss_cond, eq_cl, eq_own, eq_ctr, eq_opp

    def get_4way_values(self, i: int, j: int) -> np.ndarray:
        """Get 4-way value outputs for NN training.

        Returns array of shape (4,) with:
            [win, gam_win_cond, gam_loss_cond, bg_rate]

        Note: bg_rate is always 0 for bearoff (no backgammons possible).
        """
        win, gam_win_cond, gam_loss_cond, _, _, _, _ = self.lookup(i, j)
        return np.array([win, gam_win_cond, gam_loss_cond, 0.0], dtype=np.float32)

    def get_unconditional_gammons(self, i: int, j: int) -> Tuple[float, float]:
        """Get unconditional gammon probabilities.

        Returns:
            (gam_win, gam_loss): P(win gammon), P(lose gammon)
        """
        win, gam_win_cond, gam_loss_cond, _, _, _, _ = self.lookup(i, j)
        gam_win = win * gam_win_cond
        gam_loss = (1.0 - win) * gam_loss_cond
        return gam_win, gam_loss

    def probs_for(self, x_idx: int, o_idx: int) -> np.ndarray:
        """Return probs array for compatibility with older BearoffLookup.

        Returns 7-element array: [win, gam_win, loss, gam_loss, eq_center, eq_owner, eq_opponent]
        """
        win, gam_win_cond, gam_loss_cond, eq_cl, eq_own, eq_ctr, eq_opp = self.lookup(x_idx, o_idx)
        loss = 1.0 - win
        gam_win = win * gam_win_cond
        gam_loss = loss * gam_loss_cond
        return np.array([win, gam_win, loss, gam_loss, eq_ctr, eq_own, eq_opp], dtype=np.float32)

    def probs_for_player(self, x_idx: int, o_idx: int, cur_player: int) -> np.ndarray:
        """Return probs from the perspective of cur_player (0=X, 1=O)."""
        if cur_player == 0:
            return self.probs_for(x_idx, o_idx)
        return self.probs_for(o_idx, x_idx)

    def cube_equities(self, x_idx: int, o_idx: int, cur_player: int) -> Tuple[float, float, float]:
        """Return (centered, owner, opponent) cube equities."""
        if cur_player == 0:
            _, _, _, _, eq_own, eq_ctr, eq_opp = self.lookup(x_idx, o_idx)
        else:
            _, _, _, _, eq_own, eq_ctr, eq_opp = self.lookup(o_idx, x_idx)
        return eq_ctr, eq_own, eq_opp

    def memory_usage_bytes(self) -> int:
        """Return approximate memory usage in bytes."""
        return self.data.nbytes


def load_v4_bearoff(
    data_path: Union[str, Path],
    header_path: Union[str, Path],
) -> V4BearoffLookup:
    """Load V4 bearoff table from disk.

    Args:
        data_path: Path to binary data file (.bin)
        header_path: Path to JSON header file

    Returns:
        V4BearoffLookup instance
    """
    return V4BearoffLookup.load(data_path, header_path)
