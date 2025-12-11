# Bearoff Tables (Rust)

This package uses the Rust generator in `rust/bearoff` (Pyo3 bindings) to
reproduce gnubg's two-sided cubeful bearoff equities.

## Formats

### Version 1: Packed Upper Triangle (Legacy)
Single-perspective storage with mirroring. Has a limitation: cannot correctly
represent both (i,j) and (j,i) perspectives when they differ.

### Version 2: Tiered Dual-Perspective (Recommended)
Stores BOTH perspectives for each upper-triangle entry:
- **Tier 1** (no gammons): Positions where both players have borne off at least 1 checker
- **Tier 2** (gammons possible): At least one player has all checkers on board

Layout per entry:
- Tier 1: 10 values (5 per perspective) = 30 bytes
  `[win_ij, eq_cl_ij, eq_own_ij, eq_ctr_ij, eq_opp_ij, win_ji, ...]`
- Tier 2: 14 values (7 per perspective) = 42 bytes
  `[win_ij, gam_win_ij, gam_loss_ij, eq_cl_ij, eq_own_ij, eq_ctr_ij, eq_opp_ij, win_ji, ...]`

All values stored as uint24 (3 bytes, little endian).

## Storage Sizes (15 checkers)

| Format | Disk Size | Memory (FLOAT32) | Memory (UINT16) |
|--------|-----------|------------------|-----------------|
| V1 (broken) | ~31 GB | N/A | N/A |
| V2 Tiered | ~57 GB | ~77 GB | ~38 GB |

## Build Python bindings
```bash
cd rust/bearoff
maturin develop --release
```

## Generate v2 tiered table
```python
from bgai.endgame.packed_table import generate_and_save_tiered

tier1, tier2, header = generate_and_save_tiered(
    max_checkers=15,
    output_dir="data/tiered",
)
```

## Load with configurable precision
```python
from bgai.endgame.packed_table import load_tiered_bearoff, Precision

# Full precision (fastest lookup, highest memory)
lookup = load_tiered_bearoff(tier1_path, tier2_path, header_path, Precision.FLOAT32)

# Reduced precision (moderate memory, decode on lookup)
lookup = load_tiered_bearoff(tier1_path, tier2_path, header_path, Precision.UINT16)

# Raw storage (lowest memory, decode on lookup)
lookup = load_tiered_bearoff(tier1_path, tier2_path, header_path, Precision.UINT24)

# Lookup returns: (win, gammon_win, gammon_loss, eq_cubeless, eq_owner, eq_center, eq_opponent)
result = lookup.lookup(x_idx, o_idx)

# Memory usage
print(f"Memory: {lookup.memory_usage_bytes() / 1e9:.2f} GB")
```

## Validation
- `max_checkers=6`: matches `gnubg_ts0.bd` within 1 LSB of gnubg's uint16.
- `max_checkers=10`: matches `gnubg_6pip10checker.bd` within quantization error.

## Legacy v1 format (for backwards compatibility)
```python
from bearoff import generate_packed_bearoff

header_json, arr = generate_packed_bearoff(max_checkers=10)
# Returns shape (entries, 7, 3) uint8
```
