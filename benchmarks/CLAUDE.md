# benchmarks/ - Performance Benchmarking

Extensible benchmarking suite for MCTS and game environment performance.

## Entry Point

All benchmarks accessed via unified CLI:

```bash
python benchmark.py <type> [--discover|--validate|--single-batch N]
```

### Benchmark Types
- **game-env**: Pure environment step performance (baseline)
- **mcts**: Standard MCTS evaluator performance
- **stochastic-mcts**: StochasticMCTS for dice-based games

### Modes
- **--discover**: Find optimal batch sizes for hardware
- **--validate**: Compare against saved profiles
- **--single-batch N**: Test specific batch size

## Key Files

- **benchmark.py**: Unified CLI entry point
- **benchmark_common.py**: Base classes, memory tracking, profile management
- **benchmark_game_env.py**: Environment-only benchmarks
- **benchmark_mcts.py**: Standard MCTS benchmarks
- **benchmark_stochastic_mcts.py**: StochasticMCTS benchmarks

## Architecture

```python
class BaseBenchmark:
    def discover_optimal_batch()  # Find best batch size
    def validate_performance()    # Compare to profile
    def run_single_batch()        # Test one config
```

All benchmarks inherit from `BaseBenchmark` for consistent:
- Memory tracking (CUDA, Metal, CPU)
- Timing with `jax.block_until_ready()`
- Profile save/load (JSON)

## Example Usage

```bash
# Discover optimal batch sizes (saves profile)
python benchmark.py mcts --discover --memory-limit 20

# Validate current performance
python benchmark.py stochastic-mcts --validate

# Quick test specific batch
python benchmark.py game-env --single-batch 64 --duration 30
```

## Code Patterns

- Always call `jax.block_until_ready()` before timing
- Use `jax.jit` compilation before benchmark loop
- Track peak memory during runs
- Save profiles with hardware fingerprint
