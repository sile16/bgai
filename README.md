# bgai
Backgammon AI based on Alpha Zero training methodology.

## Current Status

**Training Active**: The system is currently running distributed training with:
- Model version: ~1030+
- Game collection: ~50+ games/min
- MLflow tracking: http://localhost:5000

### Recent Changes (Dec 2024)

- **Evaluation System**: Added eval worker with support for multiple opponent types:
  - `random`: Random policy baseline
  - `self_play`: Model vs itself
  - `gnubg`: GNU Backgammon at configurable strength
- **GNUBG Configurable Settings**: Ply depth, shortcuts, OSDB, and move filters
- **MLflow Integration**: Eval metrics and GNUBG settings logged to MLflow
- **Bearoff Detection Fix**: Corrected board index mapping for bearoff position detection
- **Redis-only Architecture**: Removed Ray dependency - standalone Python workers
- **Bearoff Generator**: New Rust/Pyo3 cubeful two-sided generator matching gnubg;
  legacy Python generators and bundled gnubg sources were removed.

## Features

- **AlphaZero-style Training**: Self-play game generation with MCTS policy improvement
- **Distributed Training**: Redis-based multi-node training (no Ray dependency)
- **Evaluation Framework**: Test models against random, self-play, and GNUBG opponents
- **Surprise-Weighted Sampling**: Prioritizes training on games where the model was most wrong
- **FIFO Episode Buffer**: Automatic eviction of old episodes with configurable capacity
- **Prometheus Metrics**: Real-time monitoring with dynamic worker discovery
- **Grafana Dashboard**: Pre-built dashboard for training visualization
- **MLflow Tracking**: Experiment tracking, metrics logging, and checkpoint management

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sile16/bgai.git
cd bgai

# Create virtual environment
python -m venv venv-bgai
source venv-bgai/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Start Distributed Training (Head Node)

```bash
# Start Redis (required for coordination)
redis-server --daemonize yes --requirepass bgai-password

# Start all services on head node
./scripts/start_all_head.sh
```

This starts:
- MLflow at http://localhost:5000
- Prometheus metrics at http://localhost:9090
- Grafana dashboard at http://localhost:3000
- Coordinator for worker management
- Training worker (GPU)
- Game worker (GPU)
- Eval worker (CPU)

### Add Remote Workers

On remote machines:
```bash
# Start a game worker (connects via Redis)
./scripts/start_game_worker.sh

# Or manually:
python -m distributed.cli.main game-worker \
    --config-file configs/distributed.yaml \
    --worker-id "remote-gpu-1"
```

### Monitor Training

- **MLflow**: http://HEAD_IP:5000 (training history, metrics, checkpoints)
- **Grafana**: http://HEAD_IP:3000 (real-time cluster stats)
- **Prometheus**: http://HEAD_IP:9090 (raw metrics)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Head Node                               │
│  ┌───────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │Coordinator│  │Training      │  │Game Worker         │    │
│  │           │  │Worker (GPU)  │  │(GPU)               │    │
│  └─────┬─────┘  └──────┬───────┘  └─────────┬──────────┘    │
│        │               │                     │               │
│        └───────────────┴─────────────────────┘               │
│                        │                                     │
│  ┌─────────┐     ┌────▼────┐     ┌─────────┐                │
│  │ MLflow  │     │  Redis  │     │  Eval   │                │
│  │ Server  │     │ Buffer  │     │ Worker  │                │
│  └─────────┘     └────┬────┘     └─────────┘                │
└────────────────────────┼────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │ Remote  │    │ Remote  │    │ Remote  │
    │ Game    │    │ Game    │    │ Game    │
    │ Worker  │    │ Worker  │    │ Worker  │
    └─────────┘    └─────────┘    └─────────┘
```

## Configuration

All settings in `configs/distributed.yaml`:

### Training Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `mcts.simulations` | 100 | MCTS simulations per move |
| `mcts.max_nodes` | 400 | Maximum tree nodes |
| `training.batch_size` | 128 | Training batch size |
| `training.games_per_batch` | 10 | Games to trigger training |
| `training.steps_per_game` | 10 | Training steps per game |
| `training.surprise_weight` | 0.5 | Surprise sampling weight |

### GNUBG Evaluation Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `gnubg.ply` | 2 | Search depth (0=fast, 2=strong) |
| `gnubg.shortcuts` | 0 | Eval shortcuts (0=off, 1=on) |
| `gnubg.osdb` | 1 | One-sided bearoff DB |
| `gnubg.move_filters` | [8,4,2,2] | Candidates per ply level |

## Evaluation System

The eval worker tests model strength against various opponents:

```bash
# Run evaluation with all opponent types
python -m distributed.cli.main eval-worker \
    --config-file configs/distributed.yaml \
    --eval-types "random,self_play,gnubg" \
    --eval-games 100
```

Results are logged to:
- Redis: `bgai:eval:results:*`
- MLflow: `eval_<type>_win_rate`, etc.
- Prometheus: `bgai_eval_*` metrics

## Prometheus Metrics

Workers expose metrics for monitoring:

| Metric | Description |
|--------|-------------|
| `bgai_games_total` | Total games completed |
| `bgai_training_loss` | Current training loss |
| `bgai_model_version` | Current model version |
| `bgai_buffer_size` | Replay buffer size |
| `bgai_games_per_minute` | Game generation rate |
| `bgai_training_steps_per_second` | Training throughput |
| `bgai_cpu_percent` | Worker CPU usage |
| `bgai_gpu_utilization_percent` | Worker GPU usage |

## Development

### Benchmarks

```bash
# Discover optimal batch sizes
python benchmark.py mcts --discover
python benchmark.py stochastic-mcts --discover
python benchmark.py game-env --discover

# Validate performance
python benchmark.py mcts --validate
```

### Testing

```bash
pytest tests/
```

### Project Structure

```
bgai/
├── bgai/               # Core backgammon environment
│   ├── gnubg_evaluator.py  # GNU Backgammon integration
│   └── endgame/        # Bearoff tables
├── core/               # TurboZero core (MCTS, training)
├── distributed/        # Distributed training system
│   ├── buffer/         # Redis replay buffer
│   ├── workers/        # Game, training, eval workers
│   ├── coordinator/    # Worker management
│   ├── cli/            # Command-line interface
│   └── metrics.py      # Prometheus metrics
├── configs/            # YAML configuration files
├── benchmarks/         # Performance benchmarks
├── scripts/            # Startup scripts
├── tools/              # Grafana, Prometheus configs
└── notebooks/          # Training notebooks
```

## Next Steps

1. **Performance Tuning**: Optimize GNUBG evaluation speed (currently CPU-bound)
2. **Extended Evaluation**: Add Elo rating estimation from eval results
3. **Checkpoint Management**: Auto-select best checkpoint based on eval metrics
4. **Multi-GPU Training**: Scale training across multiple GPUs

## License

MIT License
