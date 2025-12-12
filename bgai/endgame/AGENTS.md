# bgai/endgame/ - Bearoff Database & Endgame Tables

Perfect play database for backgammon bearoff positions.

## Overview

This module provides exact win/gammon probabilities and cube equities for bearoff positions (all checkers on home board points 1-6). Tables are generated via dynamic programming in Rust, matching GNU Backgammon's methodology.

## File Structure

| File | Purpose |
|------|---------|
| `indexing.py` | Position indexing using combinatorial number system |
| `packed_table.py` | Python lookup classes for all table formats (v1, v2, v4) |
| `validate_precomputed.py` | Validation against GNU Backgammon |
| `gnubg-1.08.003/` | GNU Backgammon source (reference only) |

## Table Formats

### Version 4 (Recommended) - `data/bearoff_*_v4.bin`

Most efficient format with conditional gammon probabilities.

**Layout**: 12 uint16 values per entry (24 bytes):
```
[gam_win_cond_ij, gam_loss_cond_ij, eq_cl_ij, eq_own_ij, eq_ctr_ij, eq_opp_ij,
 gam_win_cond_ji, gam_loss_cond_ji, eq_cl_ji, eq_own_ji, eq_ctr_ji, eq_opp_ji]
```

**Key properties**:
- Win probability derived from cubeless equity: `win = (eq_cl + 1) / 2`
- Gammons are conditional: `P(gammon|win)` and `P(gammon|loss)`
- Both perspectives (i,j) and (j,i) stored together
- Upper triangle storage: `n_entries = n_positions * (n_positions + 1) / 2`

**Decoding**:
```python
# Probability: uint16 -> [0, 1]
prob = raw_u16 / 65535.0

# Equity: uint16 -> [-1, 1]
equity = (raw_u16 / 65535.0) * 2.0 - 1.0

# Win from cubeless equity
win = (eq_cl + 1.0) / 2.0

# Unconditional gammon probs
gam_win = win * gam_win_cond
gam_loss = (1 - win) * gam_loss_cond
```

### Version 2 (Tiered) - `*_tier1.npy`, `*_tier2.npy`

Tiered storage with uint24 precision, separating gammon-possible positions.

- **Tier 1**: No gammons possible (both players borne off >=1 checker). 10 values.
- **Tier 2**: Gammons possible (one player has all checkers). 14 values.

### Version 1 (Legacy) - Packed upper triangle

7 values per entry: `[win, gam_win, loss, gam_loss, eq_center, eq_owner, eq_opponent]`

## Position Indexing

Uses GNU Backgammon's combinatorial number system for O(1) perfect hashing.

```python
from bgai.endgame.indexing import position_to_index_lut, TOTAL_ONE_SIDED_POSITIONS

# Position: tuple of 6 integers (checkers on points 1-6)
pos = (3, 2, 2, 2, 3, 3)  # 15 checkers total
idx = position_to_index_lut(np.array([pos]))[0]

# Total positions for 15 checkers on 6 points: C(21, 6) = 54,264
```

**Index methods** (fastest to slowest):
1. `position_to_index_lut()` - Precomputed LUT, ~6x faster
2. `position_to_index_batch_np()` - Vectorized numpy
3. `position_to_index_jax()` - JAX-compatible, JIT-able
4. `position_to_index()` - Pure Python reference

## Table Sizes

| Checkers | One-sided | Two-sided Entries | V4 Size |
|----------|-----------|-------------------|---------|
| 7 | 1,716 | 1,473,186 | 35 MB |
| 10 | 8,008 | 32,068,036 | 770 MB |
| 15 | 54,264 | 1,472,317,980 | 35 GB |

## Usage

### V4 Lookup (Recommended)

```python
from bgai.endgame.packed_table import V4BearoffLookup

# Load table
lookup = V4BearoffLookup.load("data/bearoff_15_v4.bin", "data/bearoff_15_v4.json")

# Look up position (i = player to move, j = opponent)
result = lookup.lookup(x_idx, o_idx)
# Returns: (win, gam_win_cond, gam_loss_cond, eq_cl, eq_own, eq_ctr, eq_opp)

# Get 4-way value outputs for NN training
value_4way = lookup.get_4way_values(x_idx, o_idx)
# Returns: [win, gam_win_cond, gam_loss_cond, bg_rate]
```

### Legacy Lookup

```python
from bgai.endgame.packed_table import load_tiered_bearoff, Precision

# Tiered format (v2)
lookup = load_tiered_bearoff(
    "data/bearoff_tiered_7_tier1.npy",
    "data/bearoff_tiered_7_tier2.npy",
    "data/bearoff_tiered_7.json",
    precision=Precision.FLOAT32
)
result = lookup.lookup(i, j)
```

## Rust Generator

Source: `rust/bearoff/src/lib.rs`

### Building

```bash
cd rust/bearoff
cargo build --release
# Produces: target/release/libbearoff.so
```

### Python Interface

```python
import sys
sys.path.insert(0, "rust/bearoff/target/release")
import bearoff

# Generate v4 format (streams to disk)
bearoff.generate_streaming_bearoff_u16(
    max_checkers=15,
    output_path="data/bearoff_15_v4.bin",
    header_path="data/bearoff_15_v4.json"
)

# Generate tiered format (in-memory)
header_json, tier1_arr, tier2_arr = bearoff.generate_tiered_bearoff(max_checkers=7)
```

## Cube Equity Values

Each entry stores four equity values for cube decisions:

| Value | Description |
|-------|-------------|
| `eq_cl` | Cubeless equity (no doubling) |
| `eq_own` | Equity when we own the cube |
| `eq_ctr` | Equity at cube in center |
| `eq_opp` | Equity when opponent owns cube |

Cube decision thresholds:
- Double if `eq_ctr >= eq_own / 2` and `eq_ctr >= eq_opp`
- Take if `eq_opp >= eq_ctr / 2`
- Pass if `eq_opp < eq_ctr / 2`

## Integration with Training

During training, bearoff positions use perfect table values instead of game outcomes:

```python
# In training_worker.py
if is_bearoff_position(board):
    # Use perfect value from table
    target = bearoff_lookup.get_4way_values(x_idx, o_idx)
else:
    # Use game outcome (standard AlphaZero)
    target = game_reward
```

See `docs/endgame_tables_design.md` for full integration design.

## Validation

```bash
# Validate against GNU Backgammon
python bgai/endgame/validate_precomputed.py
```

Validates that our generated tables match GNUBG's win probabilities exactly (bit-for-bit for uint16).
