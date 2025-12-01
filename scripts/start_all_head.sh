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
GAME_BATCH_SIZE=64
TRAIN_BATCH_SIZE=256
MCTS_SIMULATIONS=200
MCTS_MAX_NODES=800
LEARNING_RATE="3e-4"
MIN_BUFFER_SIZE=1000
WEIGHT_PUSH_INTERVAL=10
CHECKPOINT_INTERVAL=1000

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
# Start Ray head node
# =============================================================================
echo "[2/4] Starting Ray head node..."
ray start --head \
    --port="$RAY_PORT" \
    --ray-client-server-port="$RAY_CLIENT_PORT" \
    --dashboard-host=0.0.0.0 \
    --node-ip-address="$HEAD_IP" \
    --num-cpus=16 \
    --num-gpus=1

echo "       Ray dashboard: http://$HEAD_IP:8265"
echo "       Workers connect to: ray://$HEAD_IP:$RAY_CLIENT_PORT"
sleep 3

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
# Start Training Worker
# =============================================================================
echo "[4/4] Starting training worker..."
python -m distributed.cli.main training-worker \
    --coordinator-address "ray://$HEAD_IP:$RAY_CLIENT_PORT" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --min-buffer-size "$MIN_BUFFER_SIZE" \
    --weight-push-interval "$WEIGHT_PUSH_INTERVAL" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --redis-host "$REDIS_HOST" \
    --redis-port "$REDIS_PORT" \
    --redis-password "$REDIS_PASSWORD" \
    > "$LOG_DIR/training_$TIMESTAMP.log" 2>&1 &
TRAIN_PID=$!
echo "       Training worker PID: $TRAIN_PID"
echo "       Log: $LOG_DIR/training_$TIMESTAMP.log"

# =============================================================================
# Start GPU Game Worker
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
echo "  Coordinator:     $COORD_PID"
echo "  Training worker: $TRAIN_PID"
echo "  Game worker:     $GAME_PID"
echo ""
echo "Logs:"
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
