# bgai

Backgammon AI using AlphaZero-style training with distributed self-play.

## Features

- **AlphaZero-style Training**: Self-play game generation with MCTS policy improvement
- **Distributed Training**: Redis-based multi-node training (no Ray dependency)
- **Evaluation Framework**: Test models against random, self-play, and GNUBG opponents
- **Surprise-Weighted Sampling**: Prioritizes training on games where the model was most wrong
- **Real-time Monitoring**: Grafana dashboards, Prometheus metrics, MLflow tracking

## Quick Start

### Installation

```bash
git clone https://github.com/sile16/bgai.git
cd bgai
python -m venv venv-bgai
source venv-bgai/bin/activate
pip install -r requirements.txt
```

### Start Training

```bash
# Start Redis (required)
redis-server --daemonize yes --requirepass bgai-password

# Start all services
./scripts/start_all_head.sh
```

This starts:
- Coordinator for worker management
- Training worker (GPU)
- Game worker (GPU)
- Eval worker (CPU)
- MLflow at http://localhost:5000
- Grafana at http://localhost:3000
- Prometheus at http://localhost:9090

### Add Remote Workers

On remote machines:
```bash
./scripts/start_game_worker.sh
```

### Monitor Training

| Dashboard | URL | Purpose |
|-----------|-----|---------|
| Grafana | http://HEAD_IP:3000 | Real-time cluster stats |
| MLflow | http://HEAD_IP:5000 | Training history, checkpoints |
| Prometheus | http://HEAD_IP:9090 | Raw metrics |

## Architecture

```
Head Node                          Remote Workers
┌─────────────────────────┐       ┌──────────────────┐
│ Coordinator             │       │ Game Worker      │
│ Training Worker (GPU)   │◄──────┤ (self-play)      │
│ Game Worker (GPU)       │       └──────────────────┘
│ Eval Worker (CPU)       │       ┌──────────────────┐
│ Redis Buffer            │◄──────┤ Game Worker      │
│ MLflow, Grafana, Prom   │       │ (another node)   │
└─────────────────────────┘       └──────────────────┘
```

## Configuration

Edit `configs/distributed.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mcts.collect_simulations` | 200 | MCTS sims per move |
| `mcts.max_nodes` | 2000 | Maximum tree nodes |
| `training.games_per_epoch` | 200 | Games before training |
| `training.surprise_weight` | 0.5 | Surprise sampling weight |

## Evaluation

Test model strength against various opponents:

```bash
python -m distributed.cli.main eval-worker \
    --eval-types "random,self_play,gnubg" \
    --eval-games 100
```

## Development

```bash
# Run benchmarks
python benchmark.py mcts --discover
python benchmark.py stochastic-mcts --validate

# Run tests
pytest tests/
```

## Project Structure

```
bgai/
├── bgai/           # Backgammon environment
├── distributed/    # Distributed training system
├── benchmarks/     # Performance benchmarks
├── configs/        # Configuration files
├── scripts/        # Startup scripts
├── tools/          # Grafana, Prometheus
└── tests/          # Test suite
```

For detailed documentation:
- Agent guidance: `CLAUDE.md`
- Component docs: `*/CLAUDE.md`

## License

MIT License
