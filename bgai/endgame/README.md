# Bearoff Tables (Rust)

This package now uses the Rust generator in `rust/bearoff` (Pyo3 bindings) to
reproduce gnubg's two‑sided cubeful bearoff equities. The legacy Python/Numpy
generators and bundled gnubg sources have been removed.

## What is generated
- gnubg `PositionBearoff` ordering (6-point bearoff).
- Exact gnubg cube logic (`CubeEquity` from `makebearoff.c`).
- Full matrix `(n_positions, n_positions, 7)` as float32, where:
  1. `win`
  2. `gammon_win` (0.0 placeholder)
  3. `loss`
  4. `gammon_loss` (0.0 placeholder)
  5. `eq_center` (centered cube equity)
  6. `eq_owner` (equity when X owns the cube)
  7. `eq_opponent` (equity when O owns the cube)

Current build stores the full matrix; add upper-triangle packing if you want to
halve generation and disk space (see `pack_upper` in `bgai.endgame.packed_table`).

## Build Python bindings
```bash
cd rust/bearoff
maturin develop --release  # installs `bearoff` module into the active venv
```

## Generate a table (packed upper triangle, uint24 fixed)
```bash
python - <<'PY'
import numpy as np
from bearoff import generate_packed_bearoff

header_json, flat = generate_packed_bearoff(
    max_checkers=10,   # per side, 6 points
    tolerance=0.0,     # kept for API compatibility (unused)
    max_iter=0,        # kept for API compatibility (unused)
)
header = header_json
arr = np.array(flat, copy=False)  # shape (entries, 7, 3) uint8 (uint24 fixed)
np.save("data/bearoff_ts6x10_cubeful.npy", arr)
print(header)
print("saved", arr.shape, arr.dtype)
PY
```

## Validation (done)
- `max_checkers=6`: matches `gnubg_ts0.bd` within 1 LSB of gnubg's uint16.
- `max_checkers=10`: matches `gnubg_6pip10checker.bd` within quantization error.

## CLI helper
```
# generate (writes .npy and .json header)
python scripts/bearoff_cli.py generate --max-checkers 10 --output data/bearoff_ts6x10_packed_u24.npy

# show header (reads .json if present, otherwise infers basics)
python scripts/bearoff_cli.py show-header data/bearoff_ts6x10_packed_u24.npy
```

## Loading in training
Use `BearoffLookup` from `bgai.endgame.packed_table` (handles full or packed
tables, 4- or 7-slot layout). The training worker already understands the new
format; point `bearoff_table_path` at your generated `.npy` file.
