# bearoff (Rust)

Rust bearoff generator with cubeless and cubeful equities plus Python bindings.

## Build

This crate uses `pyo3` with `abi3` (Python 3.9+). Build a wheel or dev extension:

```bash
cd rust/bearoff
maturin develop --release  # or maturin build --release
```

## Python API

### Version 2: Tiered Dual-Perspective (Recommended)

```python
import bearoff

# Returns (header_json, tier1_array, tier2_array)
# Stores BOTH perspectives for each upper-triangle entry
header, tier1, tier2 = bearoff.generate_tiered_bearoff(max_checkers=15)

# tier1: shape (n_tier1_entries, 30) - 10 values * 3 bytes
# tier2: shape (n_tier2_entries, 42) - 14 values * 3 bytes
```

Tier 1 layout (no gammons possible, 10 values per entry):
- `[win_ij, eq_cl_ij, eq_own_ij, eq_ctr_ij, eq_opp_ij, win_ji, eq_cl_ji, eq_own_ji, eq_ctr_ji, eq_opp_ji]`

Tier 2 layout (gammons possible, 14 values per entry):
- `[win_ij, gam_win_ij, gam_loss_ij, eq_cl_ij, eq_own_ij, eq_ctr_ij, eq_opp_ij, win_ji, gam_win_ji, gam_loss_ji, eq_cl_ji, eq_own_ji, eq_ctr_ji, eq_opp_ji]`

All values stored as uint24 (3 bytes, little endian).

### Version 1: Packed Upper Triangle (Legacy)

```python
import bearoff

# Returns (header_json, array)
# array shape: (entries, 7, 3) uint8
header, arr = bearoff.generate_packed_bearoff(
    max_checkers=15,
    tolerance=0.0,  # unused, kept for API compatibility
    max_iter=0,     # unused, kept for API compatibility
)
```

Layout per entry (7 values, uint24 each):
- `[win, gammon_win, loss, gammon_loss, eq_center, eq_owner, eq_opponent]`

Note: V1 format has a limitation - it stores only one perspective per upper-triangle entry,
which doesn't correctly represent two-sided bearoff where (i,j) and (j,i) can have different values.
