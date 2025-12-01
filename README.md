# bgai
Backgammon AI based on Alpha Zero training methodology.

## Features

- **AlphaZero-style Training**: Self-play game generation with MCTS policy improvement
- **Distributed Training**: Ray-based multi-node training with Redis coordination
- **Surprise-Weighted Sampling**: Prioritizes training on games where the model was most wrong
- **FIFO Episode Buffer**: Automatic eviction of old episodes with configurable capacity
- **Prometheus Metrics**: Real-time monitoring with dynamic worker discovery
- **Grafana Dashboard**: Pre-built dashboard for training visualization

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
- Ray head node with dashboard at http://localhost:8265
- Prometheus metrics at http://localhost:9090
- Grafana dashboard at http://localhost:3000
- Coordinator for worker management
- Training worker (GPU)
- Game worker (GPU)

### Add Remote Workers

On remote machines:
```bash
# Join the Ray cluster
ray start --address='HEAD_IP:6380'

# Start a game worker
python -m distributed.cli.main game-worker \
    --coordinator-address "ray://HEAD_IP:10001" \
    --redis-host HEAD_IP \
    --redis-port 6379 \
    --redis-password bgai-password \
    --worker-id "remote-gpu-1" \
    --batch-size 32
```

### Monitor Training

- **Ray Dashboard**: http://HEAD_IP:8265
- **Grafana**: http://HEAD_IP:3000 (default: admin/admin)
- **Prometheus**: http://HEAD_IP:9090

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
│                   ┌────▼────┐                                │
│                   │  Redis  │  ← Replay Buffer + Weights     │
│                   └────┬────┘                                │
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

Key settings in `scripts/start_all_head.sh`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GAME_BATCH_SIZE` | 16 | Parallel games per worker |
| `TRAIN_BATCH_SIZE` | 128 | Training batch size |
| `MCTS_SIMULATIONS` | 100 | MCTS simulations per move |
| `GAMES_PER_BATCH` | 10 | Games to trigger training |
| `STEPS_PER_GAME` | 10 | Training steps per game |
| `SURPRISE_WEIGHT` | 0.5 | Surprise sampling weight (0=uniform, 1=full) |
| `MIN_BUFFER_SIZE` | 500 | Min experiences before training |

## Prometheus Metrics

Workers expose metrics for monitoring:

| Metric | Description |
|--------|-------------|
| `bgai_games_total` | Total games completed |
| `bgai_training_loss` | Current training loss |
| `bgai_model_version` | Current model version |
| `bgai_buffer_size` | Replay buffer size |
| `bgai_surprise_score_max` | Max surprise score |
| `bgai_games_per_minute` | Game generation rate |
| `bgai_training_steps_per_second` | Training throughput |

## Surprise-Weighted Sampling

Games are prioritized for training based on "surprise" - how much the model's predictions differed from actual outcomes:

```
surprise_score = |mean_value_prediction - actual_game_outcome|
```

Higher surprise scores indicate games where the model was "more wrong", making them more valuable for learning. The `SURPRISE_WEIGHT` parameter controls the blend:
- `0.0`: Uniform random sampling
- `0.5`: 50% uniform + 50% surprise-weighted
- `1.0`: Fully surprise-weighted

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
├── core/               # TurboZero core (MCTS, training)
├── distributed/        # Distributed training system
│   ├── buffer/         # Redis replay buffer
│   ├── workers/        # Game and training workers
│   ├── cli/            # Command-line interface
│   └── metrics.py      # Prometheus metrics
├── benchmarks/         # Performance benchmarks
├── scripts/            # Startup scripts
├── tools/              # Grafana, Prometheus configs
└── notebooks/          # Training notebooks
```

## License

MIT License
