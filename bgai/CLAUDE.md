# bgai/ - Backgammon Environment

Backgammon-specific code built on PGX environment.

## Files

- **bgcommon.py**: Unified step function handling stochastic (dice) and deterministic (move) states
- **bgevaluators.py**: Backgammon-specific evaluators and utilities
- **gnubg_evaluator.py**: GNU Backgammon integration for evaluation baseline

## Key Concept: Stochastic Handling

Backgammon alternates between:
1. **Stochastic states**: Dice roll (random outcome)
2. **Deterministic states**: Player move (choice among legal actions)

The `backgammon_step_fn` in `bgcommon.py` uses `jax.lax.cond` to route appropriately:

```python
new_state = jax.lax.cond(
    state._is_stochastic,
    stochastic_branch,   # env.stochastic_step()
    deterministic_branch, # env.step()
    (state, action, key)
)
```

## Endgame (`endgame/`)

Bearoff database and endgame table generation:
- **indexing.py**: Position indexing for bearoff database lookup
- **packed_table.py**: Compressed endgame table format
- **gnubg-1.08.003/**: GNU Backgammon source for database generation

See `docs/endgame_tables_design.md` for design details.

## Integration with TurboZero

Uses TurboZero's StochasticMCTS evaluator:
- Handles chance nodes (dice rolls) in tree search
- Backgammon action space: 156 possible moves

## Environment Details

- Based on PGX backgammon (`pgx.backgammon`)
- State includes board position, dice, current player
- `short_game: true` starts from mid-game positions (faster training)
