# CLAUDE.md - Project Overview

Backgammon AI using AlphaZero-style training with JAX and distributed self-play.

## Project Context

This is a **single-user ML research project**. The codebase and all hosts are fully controlled by one developer. This means:
- **No backward compatibility required** - config keys and APIs can change freely
- **No fallback code** - assume correct configuration; fail fast on errors
- **No multi-tenancy concerns** - optimize for the single deployment environment

## Directory Structure

| Directory | Purpose | Details |
|-----------|---------|---------|
| `bgai/` | Backgammon environment | [bgai/AGENTS.md](bgai/AGENTS.md) |
| `distributed/` | Worker coordination | [distributed/AGENTS.md](distributed/AGENTS.md) |
| `distributed/` | ML training pipeline | [distributed/TRAINING.md](distributed/TRAINING.md) |
| `benchmarks/` | Performance benchmarking | [benchmarks/AGENTS.md](benchmarks/AGENTS.md) |
| `tests/` | Test suite | [tests/AGENTS.md](tests/AGENTS.md) |
| `tools/` | Monitoring & visualization | [tools/AGENTS.md](tools/AGENTS.md) |
| `configs/` | Configuration files | `distributed.yaml` |
| `scripts/` | Startup/management scripts | |
| `docs/` | Design documents | |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start training cluster
./scripts/head_start.sh

# Check status
./scripts/status.sh

# Stop all
./scripts/stop_all.sh
```

## Key Dependencies

- **TurboZero**: `turbozero @ git+https://github.com/sile16/turbozero.git@main`
- **PGX**: Backgammon environment
- **JAX**: GPU-accelerated compute

## Code Patterns

- Use `jax.jit` for performance-critical functions
- Vectorize with `jax.vmap` for batch processing
- Use `jax.lax.cond` for stochastic/deterministic branching
- Always `jax.block_until_ready()` before timing

## Configuration

Edit `configs/distributed.yaml` for:
- MCTS parameters (`mcts.collect_simulations`, `mcts.max_nodes`)
- Training settings (`training.games_per_epoch`, `training.learning_rate`)
- Device-specific batch sizes (`device_configs`)

## Monitoring

| Service | Port | URL |
|---------|------|-----|
| Grafana | 3000 | `http://<HEAD_IP>:3000` |
| Prometheus | 9090 | `http://<HEAD_IP>:9090` |
| MLflow | 5000 | `http://<HEAD_IP>:5000` |

## Common Tasks

**Run benchmarks:**
```bash
python benchmark.py mcts --discover
python benchmark.py stochastic-mcts --validate
```

**Add remote worker:**
```bash
./scripts/worker_start.sh game  # On remote machine
```

**Run tests:**
```bash
pytest tests/
```
