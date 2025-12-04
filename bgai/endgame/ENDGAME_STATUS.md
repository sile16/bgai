# Endgame Table Generation Status

## Current Status: COMPLETE - VALIDATED

The numba implementation is the **primary** generator and is verified against gnubg.
The position convention mapping with gnubg is **fully understood and documented**.

### Implementation Status
- Numba matches gnubg: max_diff = 5.6e-05, mean_diff = 2.3e-05 (board_t57.bd)
- gnubg comparison uses direct tuple ordering (no reversal needed)
- 15-checker table generated and saved

### Generated Table (15 checkers, 6 points)
- **File**: `/home/sile/github/bgai/data/bearoff_15.npy`
- **Size**: 11.78 GB
- **Positions**: 54,264 (one-sided)
- **Table entries**: 2,944,581,696 (~2.94 billion)
- **Generation time**: ~172 minutes (~2.9 hours)
- **Iterations to converge**: 20

### Sample Values
```
V((0,0,0,0,0,0), (0,0,0,0,0,0)) = 1.000000  # X borne off = X wins
V((1,0,0,0,0,0), (1,0,0,0,0,0)) = 1.000000  # Both on 1pt, X moves first wins
V((0,0,0,0,0,1), (0,0,0,0,0,1)) = 0.812500  # Both on 6pt, X advantage
V((1,0,0,0,0,0), (0,0,0,0,0,1)) = 1.000000  # X easy, O hard - X wins
V((0,0,0,0,0,1), (1,0,0,0,0,0)) = 0.750000  # X hard, O easy
V((2,2,2,2,2,5), (2,2,2,2,2,5)) = 0.615990  # Full 15 checkers each
```

---

## Position Convention Mapping (UPDATED)

Our gnubg reference file `endgame/gnubg/board_t57.bd` (5 points, 7 checkers) uses
the **same tuple ordering** as our code: `tuple[0]` = 1-point (nearest bear-off),
`tuple[-1]` = furthest point. The comparison uses positions directly with no
tuple reversal and matches gnubg within 5.6e-05.

---

## Implementation Details

### Value Function
`V(X, O)` = P(X wins | X to move, optimal play)

### Dynamic Programming Formula
```
V(X, O) = 1 - E_dice[ min_{X'} V(O, X') ]
```

Where:
- `E_dice` = expectation over all 21 unique dice outcomes (weighted by probability)
- `min_{X'}` = minimum over ALL legal moves for X (optimal play)
- `V(O, X')` = opponent's win probability after X moves to X'

### Dice Outcomes
- 6 doubles: (1,1), (2,2), ..., (6,6) - each with probability 1/36
- 15 non-doubles: (1,2), (1,3), ..., (5,6) - each with probability 2/36
- Total: 21 unique outcomes

---

## Implementations

### 1. generator_numba.py (Production - PRIMARY)
- JIT-compiled with `@jit(nopython=True, parallel=True)`
- Enumerates ALL legal moves (correct minimax)
- **Verified vs gnubg**: max_diff = 5.6e-05, mean_diff = 2.3e-05 (board_t57.bd)
- ~3 hours for 15 checkers (historical run)

### 2. generator_jax.py (Experimental)
- JAX batched operations
- Incomplete - still uses greedy heuristics

---

## Performance

| Scenario | Positions (one-sided) | Time (M1) | Notes |
|----------|-----------------------|-----------|-------|
| 6 points, 8 checkers | 3,003 | ~14s | Full minimax, numba |
| 6 points, 10 checkers | 8,008 | ~216s | Full minimax, numba |
| 6 points, 15 checkers | 54,264 | ~3h (historical) | Full minimax, numba |
| 5 points, 7 checkers (gnubg match) | 792 | ~3.5s | Full minimax, numba |

---

## Files Overview

```
bgai/endgame/
├── __init__.py
├── indexing.py          # Combinatorial position indexing
├── generator.py         # Original generator (incomplete)
├── generator_numba.py   # Production implementation (fast, correct)
├── generator_jax.py     # JAX batched (experimental, incomplete)
├── generator_lowmem.py  # Memory-mapped version
├── gnubg_reader.py      # Read gnubg .bd files
├── lookup.py            # Lookup interface for training
└── gnubg/
    └── board_t57.bd     # gnubg reference database (5 points, 7 checkers)

data/
└── bearoff_15.npy       # Generated table (11.78 GB)
```

---

## Next Steps

### Immediate
1. **Fix gnubg_reader.py**: Update `compare_with_gnubg()` to use correct tuple reversal mapping
2. **Run full validation**: Compare all positions in board_t57.bd with correct mapping
3. **Document in code**: Add comments explaining convention difference

### Integration
4. **Create lookup interface**: Fast position→probability queries for training
5. **Integrate with TrainingWorker**: Use table for endgame evaluation during training
6. **PGX board conversion**: Write function to extract bearoff positions from PGX state

### Optimization (Optional)
7. **Storage optimization**: Consider float16 (6 GB instead of 12 GB)
8. **Memory-mapped access**: For systems with limited RAM

---

## Usage Example

```python
import numpy as np
from bgai.endgame.indexing import position_to_index

# Load table
table = np.load('/home/sile/github/bgai/data/bearoff_15.npy')

# Look up win probability
# Our convention: tuple[0] = 1-point, tuple[5] = 6-point
x_pos = (3, 2, 1, 0, 0, 0)  # X has 6 checkers: 3 on 1pt, 2 on 2pt, 1 on 3pt
o_pos = (2, 2, 2, 0, 0, 0)  # O has 6 checkers: 2 each on 1pt, 2pt, 3pt

x_idx = position_to_index(x_pos)
o_idx = position_to_index(o_pos)

win_prob = table[x_idx, o_idx]
print(f"P(X wins | X to move) = {win_prob:.4f}")
```

### Converting from gnubg format
```python
def gnubg_to_our_convention(gnubg_pos):
    """Convert gnubg position tuple to our convention."""
    return tuple(reversed(gnubg_pos))

# gnubg position (6pt, 5pt, 4pt, 3pt, 2pt, 1pt)
gnubg_pos = (0, 0, 0, 1, 2, 3)  # 1 on 3pt, 2 on 2pt, 3 on 1pt (gnubg)
our_pos = gnubg_to_our_convention(gnubg_pos)  # (3, 2, 1, 0, 0, 0) in our format
```

---

*Updated: 2024-12-03*
