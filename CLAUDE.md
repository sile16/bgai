# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a backgammon AI project based on Alpha Zero training methodology. The codebase implements:

- **Core AI Components**: Located in `bgai/` with backgammon-specific environment handling and evaluators
- **Distributed Training**: Ray-based multi-node training with Redis coordination in `distributed/`
- **Benchmarking Suite**: Comprehensive performance testing tools in `benchmarks/` for MCTS and StochasticMCTS algorithms
- **Training Notebooks**: Jupyter notebooks in `notebooks/` for experimental training runs

## Architecture

The project follows a modular design:

- **Environment Integration**: Uses PGX backgammon environment with custom step functions that handle both deterministic and stochastic game states
- **MCTS Evaluators**: Two main variants - standard MCTS and StochasticMCTS that adapts to the probabilistic nature of dice rolls
- **Distributed System**: Ray actors for coordination, game workers, and training workers with Redis-backed replay buffer
- **Benchmarking Framework**: Extensible base classes for performance testing with memory tracking, batch optimization, and profile management

## Dependencies

The project depends on TurboZero framework:
```
turbozero @ git+https://github.com/sile16/turbozero.git@main
```

Install with: `pip install -r requirements.txt`

## Development Commands

### Unified Benchmark CLI

All benchmarks are now accessed through a single entry point with consistent commands:

**Discovery Mode** - Find optimal batch sizes for your hardware:
```bash
# Discover optimal batch sizes for MCTS
python benchmark.py mcts --discover

# Discover for StochasticMCTS with custom settings
python benchmark.py stochastic-mcts --discover --memory-limit 32 --duration 60

# Discover for game environment with specific batch sizes
python benchmark.py game-env --discover --batch-sizes "1,4,8,16,32"
```

**Validation Mode** - Compare performance against existing profiles:
```bash
# Validate MCTS performance
python benchmark.py mcts --validate

# Validate StochasticMCTS with custom duration
python benchmark.py stochastic-mcts --validate --duration 120
```

**Single Batch Testing** - Test specific batch size:
```bash
# Test specific batch size for any benchmark
python benchmark.py mcts --single-batch 16 --duration 30
python benchmark.py game-env --single-batch 8
```

**Advanced Options:**
```bash
# Force overwrite existing profile
python benchmark.py mcts --discover --force

# Custom MCTS simulations
python benchmark.py mcts --discover --num-simulations 100

# Verbose output for debugging
python benchmark.py game-env --discover --verbose
```

### Benchmark Types

- **`game-env`**: Pure environment step performance (baseline)
- **`mcts`**: Standard MCTS evaluator performance  
- **`stochastic-mcts`**: StochasticMCTS evaluator for dice-based games

## Key Implementation Details

- **JAX Integration**: All core algorithms use JAX for JIT compilation and vectorization
- **Stochastic Handling**: The `bgcommon.py` contains a unified step function that conditionally handles stochastic vs deterministic states using `jax.lax.cond`
- **Memory Management**: Benchmarks include sophisticated memory tracking with device-specific reporting (CUDA, Metal, CPU)
- **Profile System**: Benchmark results are automatically saved with hardware-specific profiles for comparison and validation

## Code Patterns

- Use `jax.jit` for performance-critical functions
- Vectorize operations with `jax.vmap` for batch processing
- Handle stochastic/deterministic branching with `jax.lax.cond`
- Always call `jax.block_until_ready()` for accurate timing measurements
- Use the common benchmark framework in `benchmark_common.py` for consistent testing

## Distributed Training

### Starting the Cluster

```bash
# Start Redis first
redis-server --daemonize yes --requirepass bgai-password

# Start head node with all services
./scripts/start_all_head.sh

# Check status
./scripts/status.sh

# Stop all
./scripts/stop_all.sh
```

### Key Components

- **Coordinator** (`distributed/coordinator/`): Manages worker registration and weight distribution
- **Game Workers** (`distributed/workers/game_worker.py`): Generate self-play games with MCTS
- **Training Workers** (`distributed/workers/training_worker.py`): Train neural network from replay buffer
- **Redis Buffer** (`distributed/buffer/redis_buffer.py`): Stores experiences with FIFO eviction and surprise-weighted sampling

### Metrics & Monitoring

Workers expose Prometheus metrics on ports 9100 (game) and 9200 (training):
- Dynamic discovery via Redis registration
- Grafana dashboard at `tools/grafana_bgai_dashboard.json`
- Key metrics: `bgai_training_loss`, `bgai_games_total`, `bgai_surprise_score_*`

### Configuration

Edit `scripts/start_all_head.sh` for training parameters:
- `GAMES_PER_BATCH`: Games required to trigger training (default: 10)
- `STEPS_PER_GAME`: Training steps per collected game (default: 10)
- `SURPRISE_WEIGHT`: Blend of uniform vs surprise-weighted sampling (default: 0.5)