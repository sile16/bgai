# Endgame Tables Integration Design

## Executive Summary

**Goal**: Inject perfect endgame values into training to teach the NN perfect bearoff play.

**Key Decisions**:
1. **Training-time only** - No TurboZero/MCTS changes needed
2. **JAX-native tables** - Pre-compute once, store as numpy arrays, fast lookup
3. **Surpass gnubg** - Use gnubg to bootstrap, then our own DP for larger tables

**Architecture**:
```
Game Collection (GameWorker)     Training (TrainingWorker)
        |                                |
        v                                v
   NN evaluates moves            Load experiences from buffer
        |                                |
        v                                v
   Store experiences             For endgame positions:
   (position, policy,            - Lookup perfect value from table
    game_outcome)                - Replace game_outcome with perfect value
                                        |
                                        v
                                 Compute loss with perfect targets
                                 NN learns perfect endgame evaluation
```

---

## Research Summary

### GNU Backgammon Bearoff Databases

GNU Backgammon provides two types of bearoff databases:

1. **One-sided (approximative)**: Probability distribution of bearing off in N rolls for each player independently
   - Size: 54,264 positions for 15 checkers on 6 points
   - File: `gnubg_os0.bd`

2. **Two-sided (exact)**: Exact win probability considering BOTH players' positions
   - Size: 294M+ positions for equivalent coverage
   - File: `gnubg_ts0.bd`
   - **This is what we need** - opponent's position significantly affects optimal play

### Key Finding: Two-Sided Matters

Example demonstrating two-sided importance (same X position, different O positions):
```
X stacked on 6pt, O on 1pt: X win = 99.995%
X stacked on 1pt, O on 6pt: X win = 0.005%
```

The same X position has vastly different values depending on O's position.

### Available API (gnubg Python module)

The gnubg module already provides everything we need:

```python
import gnubg

# Position classification
gnubg.classify(board)  # Returns: c_bearoff=2, c_race=3, c_contact=5

# Exact two-sided probabilities (win, gammon, bg, opp_gammon, opp_bg)
gnubg.probabilities(board, ply=0)  # ply=0 uses bearoff database directly

# One-sided bearoff probability distribution
gnubg.bearoff_probabilities(pos_tuple)  # P(bear off in N rolls)
```

### Board Conversion

Utilities exist in `pgx/tools/gnubg_bridge.py`:
- `pgx_to_gnubg_board(pgx_board)` - Convert PGX 28-element array to gnubg format
- `gnubg_to_pgx_board(gnubg_board)` - Convert back

---

## Integration Design

### Phase 1: Perfect Value Collection (GameWorker)

Modify experience collection to detect and store perfect endgame values.

**File: `distributed/workers/game_worker.py`**

```python
def is_perfect_value_position(pgx_board: np.ndarray) -> bool:
    """Check if position has a perfectly calculated value."""
    gnubg_board = pgx_to_gnubg_board(pgx_board)
    pos_class = gnubg.classify(gnubg_board)
    # c_bearoff=2 (exact), c_race=3 (near-exact with database)
    return pos_class in (gnubg.c_bearoff, gnubg.c_race)

def get_perfect_value(pgx_board: np.ndarray) -> float:
    """Get perfect win probability for endgame position."""
    gnubg_board = pgx_to_gnubg_board(pgx_board)
    probs = gnubg.probabilities(gnubg_board, 0)  # ply=0 = database lookup
    # probs = (win, gammon, backgammon, opp_gammon, opp_backgammon)
    # Value = P(win) + P(gammon) + P(bg) - P(opp_gammon) - P(opp_bg)
    # Or simpler: just P(win) for money game
    return probs[0]  # Win probability
```

**Experience Structure Extension:**
```python
@dataclass
class Experience:
    observation: np.ndarray
    policy_weights: np.ndarray
    reward: float  # Game outcome (backpropagated)
    perfect_value: Optional[float] = None  # Perfect DB value if available
    has_perfect_value: bool = False
```

### Phase 2: Training Integration (TrainingWorker)

Use perfect values instead of game outcomes when available.

**File: `distributed/workers/training_worker.py`**

Modify loss computation:
```python
def compute_value_target(experience):
    if experience.has_perfect_value:
        # Use exact database value
        return experience.perfect_value
    else:
        # Use game outcome (standard AlphaZero)
        return experience.reward
```

### Phase 3: JAX-Compatible Implementation

Since gnubg uses Python callbacks, we need to handle this carefully:

**Option A: Pre-compute during collection** (Recommended)
- Compute perfect values in GameWorker (already uses `jax.pure_callback`)
- Store alongside experiences
- No JAX tracing issues

**Option B: Batch lookup in training**
- Use `jax.pure_callback` in loss function
- Higher overhead but more flexible

### Implementation Priority

1. **First**: Add `perfect_value` field to experience/buffer
2. **Second**: Detect and compute perfect values in GameWorker
3. **Third**: Use perfect values in TrainingWorker loss computation
4. **Fourth**: Add metrics for % of experiences with perfect values

---

## Future: Anchor Points

The same infrastructure supports "anchor points" - pre-computed values for common non-endgame positions:

1. **Position Hashing**: Use gnubg's position ID encoding or custom hash
2. **Anchor Database**: Redis hash or separate file storing position -> value
3. **Generation**:
   - Run deep MCTS (10K+ simulations) on common positions
   - Store resulting values as anchor points
4. **Usage**: Same as endgame tables - inject into training

### Candidate Anchor Positions
- Opening positions (first 5-10 moves)
- Common racing positions
- Key back-game formations
- Cube decision positions

---

## Files to Modify

1. `distributed/buffer/redis_buffer.py` - Add perfect_value field to experience
2. `distributed/workers/game_worker.py` - Detect endgames, compute perfect values
3. `distributed/workers/training_worker.py` - Use perfect values in loss
4. `bgai/endgame_evaluator.py` (new) - Endgame detection and value lookup

## Dependencies

- gnubg Python module (already installed) - for bootstrapping table generation
- numpy - for storing tables
- JAX - for JIT-compiled position indexing

---

## Detailed Implementation Plan

### Step 1: Build Position Indexing (JAX-native)

```python
# bgai/endgame/indexing.py
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(1,))
def position_to_index(checkers_per_point: jnp.ndarray, max_checkers: int) -> int:
    """Convert 6-element array of checker counts to table index.

    Uses combinatorial number system for O(1) perfect hashing.
    Fully JIT-compatible.
    """
    # Implementation using Pascal's triangle / combinatorial math
    # Returns index in range [0, C(n+5,5)-1]
    pass

def index_to_position(idx: int, max_checkers: int) -> tuple:
    """Inverse of position_to_index."""
    pass
```

### Step 2: Generate Bearoff Tables

**Option A: Bootstrap from gnubg (quick start)**
```python
# scripts/generate_bearoff_tables.py
import gnubg
import numpy as np

def generate_two_sided_table(max_checkers=6):
    """Generate exact two-sided bearoff table using gnubg."""
    n_positions = comb(max_checkers + 5, 5)
    table = np.zeros((n_positions, n_positions), dtype=np.float32)

    for x_idx, x_pos in enumerate(all_positions(max_checkers)):
        for o_idx, o_pos in enumerate(all_positions(max_checkers)):
            board = positions_to_gnubg_board(x_pos, o_pos)
            probs = gnubg.probabilities(board, 0)
            table[x_idx, o_idx] = probs[0]  # Win probability

    np.save('data/bearoff_6x6.npy', table)
```

**Option B: Native dynamic programming (faster, no gnubg dependency)**
```python
def generate_bearoff_dp(max_checkers=6):
    """Generate bearoff table via backwards induction.

    Start from terminal states (all checkers off), work backwards.
    For each state, compute expected value over all dice rolls.
    """
    # Terminal states: P(win) = 1 if X bears off first
    # For each non-terminal state:
    #   P(X wins) = sum over X's rolls of:
    #     P(roll) * max over X's moves of:
    #       sum over O's rolls of:
    #         P(roll) * max over O's moves of:
    #           P(X wins | resulting state)
    pass
```

### Step 3: Integrate into Training

**File: `distributed/workers/training_worker.py`**

```python
class TrainingWorker:
    def __init__(self, config, ...):
        # Load bearoff table
        self.bearoff_table = np.load('data/bearoff_6x6.npy')
        self.bearoff_max_checkers = 6

    def _is_bearoff_position(self, board):
        """Check if position is in bearoff database."""
        # Both players have all checkers on home board (points 1-6)
        # and total checkers <= max_checkers
        pass

    def _get_perfect_value(self, board):
        """Lookup perfect value from table."""
        x_pos = self._extract_home_board(board, player=0)
        o_pos = self._extract_home_board(board, player=1)
        x_idx = position_to_index(x_pos, self.bearoff_max_checkers)
        o_idx = position_to_index(o_pos, self.bearoff_max_checkers)
        return self.bearoff_table[x_idx, o_idx]

    def _compute_value_targets(self, experiences):
        """Compute value targets, using perfect values when available."""
        targets = []
        for exp in experiences:
            if self._is_bearoff_position(exp.observation):
                # Use perfect value
                target = self._get_perfect_value(exp.observation)
            else:
                # Use game outcome (standard AlphaZero)
                target = exp.reward
            targets.append(target)
        return np.array(targets)
```

### Step 4: Table Sizes and Coverage

| Max Checkers | One-sided Positions | Two-sided Positions | Memory |
|-------------|--------------------|--------------------|--------|
| 6           | 462                | 213,444            | 0.8 MB |
| 8           | 1,287              | 1,656,369          | 6.3 MB |
| 10          | 3,003              | 9,018,009          | 34 MB  |
| 12          | 6,188              | 38,291,344         | 146 MB |
| 15          | 15,504             | 240,374,016        | 917 MB |

**Recommendation**: Start with 8-checker (6.3 MB), expand to 10-12 if needed.

### Step 5: Future - Anchor Points for Common Positions

Same infrastructure for pre-computed mid-game positions:

1. **Identify anchor candidates**: Opening book positions, common formations
2. **Compute values**: Run deep MCTS (10K+ sims) or long rollouts
3. **Store in lookup table**: Hash position -> value
4. **Use in training**: Same as bearoff tables

---

## Why This Surpasses gnubg

1. **NN generalizes** - gnubg uses fixed neural nets + lookup; our NN learns from perfect data
2. **Larger coverage** - We can generate bigger tables than gnubg ships with
3. **Faster inference** - Pure JAX NN evaluation vs gnubg's Python callbacks
4. **Better training signal** - Perfect values reduce noise in value targets

## Sources

- [GNU Backgammon Manual](https://www.gnu.org/software/gnubg/manual/gnubg.html)
- [Bearoff Databases](https://www.gnu.org/software/gnubg/manual/html_node/Bearoff-databases-with-GNU-Backgammon.html)
- [makebearoff manual](https://www.mankier.com/6/makebearoff)
- [gnubg-nn-pypi](https://github.com/reayd-falmouth/gnubg-nn-pypi)
