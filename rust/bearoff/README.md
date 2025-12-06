# bearoff (Rust)

Rust bearoff generator with cubeless and cubeful equities plus Python bindings.

## Build

This crate uses `pyo3` with `abi3` (Python 3.9+). Build a wheel or dev extension:

```bash
cd rust/bearoff
maturin develop --release  # or maturin build --release
```

## Python API

```python
import bearoff

# Returns packed upper-triangle array of shape (entries, 4 or 7)
# entries = n*(n+1)//2 where n = total one-sided positions for max_checkers
arr = bearoff.generate_packed_bearoff(
    max_checkers=15,
    cubeful=True,       # False -> only cubeless probs
    tolerance=1e-6,
    max_iter=100,
)
```

Layout per packed row:
- cubeless: `[win, gammon_win(0), loss, gammon_loss(0)]`
- cubeful:  `[win, 0, loss, 0, eq_center, eq_owner, eq_opponent]`
