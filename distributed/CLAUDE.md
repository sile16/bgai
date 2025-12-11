# distributed/ - Worker Coordination

Redis-based multi-node coordination system. No Ray dependency.

## Architecture

```
Head Node                          Remote Workers
┌─────────────────────────┐       ┌──────────────────┐
│ Redis (coordination)    │◄──────┤ Game Worker      │
│ Coordinator             │       └──────────────────┘
│ Workers (game/train/eval)│      ┌──────────────────┐
└─────────────────────────┘◄──────┤ Game Worker      │
                                  └──────────────────┘
```

## Components

### Coordinator (`coordinator/`)
- **head_node.py**: Worker registration, weight distribution, heartbeat monitoring
- **redis_state.py**: Shared state (model version, training status, pause flags)

### Workers (`workers/`)
- **base_worker.py**: Common functionality for all workers
  - Heartbeat with TTL for health monitoring
  - Auto-registration on startup
  - Model weight sync from Redis
  - Graceful shutdown handling
- **game_worker.py**: Self-play game generation (see TRAINING.md)
- **training_worker.py**: Neural network training (see TRAINING.md)
- **eval_worker.py**: Model evaluation against baselines

### CLI (`cli/`)
- **main.py**: Entry points (`game-worker`, `training-worker`, `eval-worker`, `coordinator`)
- **config_loader.py**: YAML config with device-specific overrides

## Running

```bash
# Head node (all services)
./scripts/start_all_head.sh

# Remote game worker
./scripts/start_game_worker.sh

# Check status
./scripts/status.sh

# Stop all
./scripts/stop_all.sh
```

## Redis Keys

| Key Pattern | Purpose |
|-------------|---------|
| `bgai:workers:*` | Worker registration (TTL=30s) |
| `bgai:model:version` | Current model version |
| `bgai:model:weights` | Serialized model weights |
| `bgai:training:paused` | Collection pause flag |
| `bgai:warm_tree` | Warm MCTS tree for initialization |

## Configuration (`configs/distributed.yaml`)

```yaml
coordinator:
  heartbeat_timeout: 30.0   # Worker TTL
  heartbeat_interval: 10.0  # Heartbeat frequency
  weight_push_interval: 10  # Push weights every N steps

device_configs:
  cuda:
    game_batch_size: 128
    train_batch_size: 512
```

## Code Patterns

- Workers are standalone Python processes
- Redis pub/sub for real-time coordination
- Async operations to hide network latency
- Exponential backoff on reconnection

See also: [TRAINING.md](TRAINING.md) for ML pipeline details
