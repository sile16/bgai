# distributed/ - ML Training Pipeline

AlphaZero-style training with self-play and experience replay.

## Training Loop

```
┌─────────────┐    experiences    ┌──────────────┐    samples    ┌─────────────┐
│ Game Worker │ ─────────────────►│ Redis Buffer │ ─────────────►│  Training   │
│ (self-play) │                   │ (replay)     │               │   Worker    │
└─────────────┘                   └──────────────┘               └─────────────┘
       ▲                                                               │
       │                         new weights                           │
       └───────────────────────────────────────────────────────────────┘
```

## Game Worker (`workers/game_worker.py`)

Generates self-play games using StochasticMCTS:

- **Batched environments**: 128 parallel games (configurable)
- **MCTS policy**: 200 simulations per move
- **Temperature schedule**: 0.8 → 0.2 over 50 epochs (exploration → exploitation)
- **Async Redis sender**: Hides network latency, maintains ~90%+ GPU utilization

Experience format: `(state, policy, value, surprise_score)`

## Training Worker (`workers/training_worker.py`)

Trains neural network from replay buffer:

- **Collection-gated**: Waits for N games before training
- **Surprise-weighted sampling**: Prioritizes positions where model was wrong
- **Loss function**: Policy cross-entropy + Value MSE + L2 regularization
- **Checkpointing**: Every 5 epochs, keeps last 5

```python
# Surprise weight blends uniform and surprise sampling
sample_weight = (1 - surprise_weight) * uniform + surprise_weight * surprise
```

## Redis Buffer (`buffer/redis_buffer.py`)

Experience replay with intelligent sampling:

- **FIFO eviction**: Oldest experiences removed when buffer full
- **Surprise scoring**: `|predicted_value - actual_outcome|`
- **Episode storage**: Full game trajectories for value bootstrapping
- **Capacity**: 100k experiences, 5k episodes (configurable)

## Configuration

```yaml
training:
  batch_size: 1024
  learning_rate: 0.0003
  l2_reg_lambda: 0.0001
  games_per_epoch: 200        # Games before training
  surprise_weight: 0.5        # Sampling blend

mcts:
  collect_simulations: 200
  max_nodes: 2000
  temperature_start: 0.8
  temperature_end: 0.2
  temperature_epochs: 50
```

## Evaluation Worker (`workers/eval_worker.py`)

Tests model strength (runs on CPU):

- **random**: Win rate vs random policy
- **self_play**: Current vs previous model version
- **gnubg**: Win rate vs GNU Backgammon (configurable ply)

Results logged to MLflow and Prometheus.

## JAX Patterns

```python
# JIT compile for performance
@jax.jit
def train_step(params, batch):
    ...

# Vectorize over batch
batched_eval = jax.vmap(eval_fn)

# Handle stochastic/deterministic states
new_state = jax.lax.cond(state._is_stochastic, stoch_fn, det_fn, ...)
```
