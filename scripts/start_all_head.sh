#!/bin/bash
# Start all head node services (coordinator, training worker, game worker)
# Everything runs in background with logs to files
#
# Usage: ./scripts/start_all_head.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# =============================================================================
# Activate virtual environment
# =============================================================================
if [[ -d "$PROJECT_DIR/venv-bgai" ]]; then
    source "$PROJECT_DIR/venv-bgai/bin/activate"
elif [[ -d "$PROJECT_DIR/venv" ]]; then
    source "$PROJECT_DIR/venv/bin/activate"
else
    echo "WARNING: No virtual environment found. Using system Python."
fi
echo "Using Python: $(which python)"

# =============================================================================
# Configuration
# =============================================================================
# Try Tailscale IP first, fallback to local IP
TAILSCALE_IP="100.105.50.111"
LOCAL_IP="192.168.20.40"

if ping -c 1 -W 1 "$TAILSCALE_IP" &>/dev/null; then
    HEAD_IP="$TAILSCALE_IP"
    echo "Using Tailscale IP: $HEAD_IP"
else
    HEAD_IP="$LOCAL_IP"
    echo "Tailscale unavailable, using local IP: $HEAD_IP"
fi
REDIS_HOST="$HEAD_IP"
REDIS_PORT="6379"
REDIS_PASSWORD="bgai-password"
RAY_PORT="6380"
RAY_CLIENT_PORT="10001"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"
LOG_DIR="$PROJECT_DIR/logs"

# GPU/CUDA settings (head node)
# Note: Smaller values for faster JIT compilation on first run
# Increase after confirming system works
GAME_BATCH_SIZE=16         # Reduced from 64 for faster JIT
TRAIN_BATCH_SIZE=128       # Reduced from 256
MCTS_SIMULATIONS=100       # Reduced from 200 for faster JIT
MCTS_MAX_NODES=400         # Reduced from 800
LEARNING_RATE="3e-4"
MIN_BUFFER_SIZE=500        # Reduced from 1000 for faster first train
CHECKPOINT_INTERVAL=1000

# Collection-gated training settings
# Training triggers after GAMES_PER_BATCH new games, then runs STEPS_PER_GAME * new_games steps
GAMES_PER_BATCH=10         # New games required to trigger training batch
STEPS_PER_GAME=10          # Training steps per collected game (e.g., 10 games -> 100 steps)

# Surprise-weighted sampling (0=uniform, 1=fully surprise-weighted)
# Higher weights focus training on games where model predictions differed from actual outcome
SURPRISE_WEIGHT=0.5

# =============================================================================
# Setup
# =============================================================================
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# Timestamp for log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "  BGAI Distributed Training - Head Node"
echo "=============================================="
echo "Project dir: $PROJECT_DIR"
echo "Logs dir:    $LOG_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
echo ""

# =============================================================================
# Stop existing processes
# =============================================================================
echo "[1/4] Stopping any existing Ray cluster..."
ray stop --force 2>/dev/null || true
pkill -f "distributed.cli.main" 2>/dev/null || true
sleep 2

# =============================================================================
# Set PYTHONPATH so Ray workers can find the distributed module
# =============================================================================
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# =============================================================================
# JAX memory configuration - prevent first process from grabbing all GPU RAM
# Based on benchmarks: game worker ~2GB, training worker ~2GB
# Set to 25% each (6GB) for headroom (24GB total GPU)
# =============================================================================
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
echo "JAX memory fraction: $XLA_PYTHON_CLIENT_MEM_FRACTION (6GB per worker on 24GB GPU)"

# =============================================================================
# Start Ray head node
# =============================================================================
echo "[2/4] Starting Ray head node..."
# Bind to 0.0.0.0 to accept connections from both Tailscale and LAN
RAY_RUNTIME_ENV_CREATE_WORKING_DIR=1 ray start --head \
    --port="$RAY_PORT" \
    --ray-client-server-port="$RAY_CLIENT_PORT" \
    --dashboard-host=0.0.0.0 \
    --node-ip-address=0.0.0.0 \
    --num-cpus=16 \
    --num-gpus=1

echo "       Ray dashboard: http://$HEAD_IP:8265"
echo "       Workers connect to: ray://$HEAD_IP:$RAY_CLIENT_PORT"
sleep 3

# =============================================================================
# Start Prometheus metrics with custom BGAI config
# =============================================================================
echo "       Starting Prometheus metrics..."
PROMETHEUS_BIN="$PROJECT_DIR/prometheus-3.7.3.linux-amd64/prometheus"
PROMETHEUS_CONFIG="$PROJECT_DIR/tools/prometheus_bgai.yml"
if [[ -f "$PROMETHEUS_BIN" ]]; then
    "$PROMETHEUS_BIN" \
        --config.file "$PROMETHEUS_CONFIG" \
        --web.enable-lifecycle \
        > "$LOG_DIR/prometheus_$TIMESTAMP.log" 2>&1 &
    PROMETHEUS_PID=$!
    echo "       Prometheus PID: $PROMETHEUS_PID (custom BGAI config)"
else
    # Fallback to Ray's built-in Prometheus
    ray metrics launch-prometheus > "$LOG_DIR/prometheus_$TIMESTAMP.log" 2>&1 &
    PROMETHEUS_PID=$!
    echo "       Prometheus PID: $PROMETHEUS_PID (Ray default)"
fi
echo "       Prometheus UI: http://$HEAD_IP:9090"
sleep 2

# =============================================================================
# Start Grafana for Ray dashboard metrics visualization
# =============================================================================
GRAFANA_DIR="$PROJECT_DIR/tools/grafana-11.3.0"
if [[ -d "$GRAFANA_DIR" ]]; then
    echo "       Starting Grafana..."
    pkill -f "grafana-server" 2>/dev/null || true
    sleep 1
    "$GRAFANA_DIR/bin/grafana-server" \
        --homepath="$GRAFANA_DIR" \
        --config="$GRAFANA_DIR/conf/custom.ini" \
        > "$LOG_DIR/grafana_$TIMESTAMP.log" 2>&1 &
    GRAFANA_PID=$!
    echo "       Grafana PID: $GRAFANA_PID"
    echo "       Grafana UI: http://$HEAD_IP:3000"
    sleep 2
else
    echo "       WARNING: Grafana not installed. Run ./scripts/setup_grafana.sh first"
    GRAFANA_PID=""
fi

# =============================================================================
# Start Prometheus discovery updater (watches Redis for worker registrations)
# =============================================================================
echo "       Starting Prometheus discovery updater..."
python -c "
from distributed.metrics import start_discovery_updater
updater = start_discovery_updater(
    redis_host='$REDIS_HOST',
    redis_port=$REDIS_PORT,
    redis_password='$REDIS_PASSWORD',
    update_interval=15,
)
print('Discovery updater started')
import time
while True:
    time.sleep(3600)  # Keep running
" > "$LOG_DIR/discovery_updater_$TIMESTAMP.log" 2>&1 &
DISCOVERY_PID=$!
echo "       Discovery updater PID: $DISCOVERY_PID"
sleep 2

# =============================================================================
# Start Coordinator
# =============================================================================
echo "[3/4] Starting coordinator..."
python -m distributed.cli.main coordinator \
    --dashboard \
    --redis-host "$REDIS_HOST" \
    --redis-port "$REDIS_PORT" \
    --redis-password "$REDIS_PASSWORD" \
    --mcts-simulations "$MCTS_SIMULATIONS" \
    --mcts-max-nodes "$MCTS_MAX_NODES" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --game-batch-size "$GAME_BATCH_SIZE" \
    > "$LOG_DIR/coordinator_$TIMESTAMP.log" 2>&1 &
COORD_PID=$!
echo "       Coordinator PID: $COORD_PID"
echo "       Log: $LOG_DIR/coordinator_$TIMESTAMP.log"
sleep 5

# Check coordinator started
if ! kill -0 $COORD_PID 2>/dev/null; then
    echo "ERROR: Coordinator failed to start. Check log:"
    tail -20 "$LOG_DIR/coordinator_$TIMESTAMP.log"
    exit 1
fi

# =============================================================================
# Start Training Worker (with GPU - uses 0.5 GPU to share with game worker)
# Collection-gated: waits for new games before training
# =============================================================================
echo "[4/4] Starting training worker..."
python -m distributed.cli.main training-worker \
    --coordinator-address "ray://$HEAD_IP:$RAY_CLIENT_PORT" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --min-buffer-size "$MIN_BUFFER_SIZE" \
    --games-per-batch "$GAMES_PER_BATCH" \
    --steps-per-game "$STEPS_PER_GAME" \
    --surprise-weight "$SURPRISE_WEIGHT" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --redis-host "$REDIS_HOST" \
    --redis-port "$REDIS_PORT" \
    --redis-password "$REDIS_PASSWORD" \
    --num-gpus 0.5 \
    > "$LOG_DIR/training_$TIMESTAMP.log" 2>&1 &
TRAIN_PID=$!
echo "       Training worker PID: $TRAIN_PID"
echo "       Log: $LOG_DIR/training_$TIMESTAMP.log"

# =============================================================================
# Start GPU Game Worker (uses 0.5 GPU to share with training worker)
# =============================================================================
echo "[5/5] Starting GPU game worker..."
python -m distributed.cli.main game-worker \
    --coordinator-address "ray://$HEAD_IP:$RAY_CLIENT_PORT" \
    --worker-id "gpu-head" \
    --batch-size "$GAME_BATCH_SIZE" \
    --mcts-simulations "$MCTS_SIMULATIONS" \
    --mcts-max-nodes "$MCTS_MAX_NODES" \
    --temperature 1.0 \
    --redis-host "$REDIS_HOST" \
    --redis-port "$REDIS_PORT" \
    --redis-password "$REDIS_PASSWORD" \
    --num-gpus 0.5 \
    > "$LOG_DIR/game_gpu_$TIMESTAMP.log" 2>&1 &
GAME_PID=$!
echo "       Game worker PID: $GAME_PID"
echo "       Log: $LOG_DIR/game_gpu_$TIMESTAMP.log"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "  All services started!"
echo "=============================================="
echo ""
echo "PIDs:"
echo "  Prometheus:      $PROMETHEUS_PID"
if [[ -n "$GRAFANA_PID" ]]; then
echo "  Grafana:         $GRAFANA_PID"
fi
echo "  Coordinator:     $COORD_PID"
echo "  Training worker: $TRAIN_PID"
echo "  Game worker:     $GAME_PID"
echo ""
echo "Dashboards:"
echo "  Ray Dashboard:   http://$HEAD_IP:8265"
echo "  Grafana:         http://$HEAD_IP:3000"
echo "  Prometheus:      http://$HEAD_IP:9090"
echo ""
echo "Logs:"
echo "  tail -f $LOG_DIR/prometheus_$TIMESTAMP.log"
if [[ -n "$GRAFANA_PID" ]]; then
echo "  tail -f $LOG_DIR/grafana_$TIMESTAMP.log"
fi
echo "  tail -f $LOG_DIR/coordinator_$TIMESTAMP.log"
echo "  tail -f $LOG_DIR/training_$TIMESTAMP.log"
echo "  tail -f $LOG_DIR/game_gpu_$TIMESTAMP.log"
echo ""
echo "Monitor all logs:"
echo "  tail -f $LOG_DIR/*_$TIMESTAMP.log"
echo ""
echo "Check status:"
echo "  ./scripts/status.sh"
echo ""
echo "Stop all:"
echo "  ./scripts/stop_all.sh"
echo ""
echo "Remote workers connect with:"
echo "  ray://$HEAD_IP:$RAY_CLIENT_PORT"
echo "  Redis: $REDIS_HOST:$REDIS_PORT"
