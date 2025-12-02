#!/bin/bash
# Start a game worker with auto-detection of platform (Mac/Linux+CUDA/TPU)
# Runs in background with logs to file
#
# Usage: ./scripts/start_game_worker.sh [worker_id]

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
# Configuration - Head node connection
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
LOG_DIR="$PROJECT_DIR/logs"

# =============================================================================
# Join Ray cluster as worker node (for distributed resources)
# =============================================================================
join_ray_cluster() {
    # Check if already connected to a Ray cluster
    if ray status &>/dev/null; then
        echo "Already connected to Ray cluster"
        return 0
    fi

    # Enable multi-node clusters on Mac/Windows
    export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1

    echo "Joining Ray cluster at $HEAD_IP:$RAY_PORT..."
    ray start --address="$HEAD_IP:$RAY_PORT" --block &
    RAY_PID=$!

    # Wait for connection
    for i in {1..30}; do
        if ray status &>/dev/null; then
            echo "Successfully joined Ray cluster"
            return 0
        fi
        sleep 1
    done

    echo "ERROR: Failed to join Ray cluster after 30 seconds"
    return 1
}

# =============================================================================
# Auto-detect platform and set appropriate parameters
# =============================================================================
detect_platform() {
    local os_type=$(uname -s)
    local has_cuda=false
    local has_tpu=false

    # Check for CUDA
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        has_cuda=true
    fi

    # Check for TPU (future support)
    if [[ -d "/dev/accel" ]] || [[ -n "${TPU_NAME:-}" ]]; then
        has_tpu=true
    fi

    if [[ "$os_type" == "Darwin" ]]; then
        echo "mac"
    elif $has_tpu; then
        echo "tpu"
    elif $has_cuda; then
        echo "cuda"
    else
        echo "cpu"
    fi
}

PLATFORM=$(detect_platform)
echo "Detected platform: $PLATFORM"

# Set platform-specific parameters
case "$PLATFORM" in
    cuda)
        BATCH_SIZE=64
        MCTS_SIMULATIONS=200
        MCTS_MAX_NODES=800
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        PLATFORM_TAG="cuda"
        ;;
    mac)
        BATCH_SIZE=16
        MCTS_SIMULATIONS=100
        MCTS_MAX_NODES=400
        export JAX_PLATFORMS="cpu"
        PLATFORM_TAG="mac"
        ;;
    tpu)
        # Future TPU support
        BATCH_SIZE=128
        MCTS_SIMULATIONS=200
        MCTS_MAX_NODES=800
        PLATFORM_TAG="tpu"
        ;;
    cpu|*)
        BATCH_SIZE=8
        MCTS_SIMULATIONS=50
        MCTS_MAX_NODES=200
        export JAX_PLATFORMS="cpu"
        PLATFORM_TAG="cpu"
        ;;
esac

# =============================================================================
# Worker ID
# =============================================================================
if [[ -n "$1" ]]; then
    WORKER_ID="$1"
else
    HOSTNAME_SHORT=$(hostname -s 2>/dev/null || hostname)
    WORKER_ID="${PLATFORM_TAG}-${HOSTNAME_SHORT}-$$"
fi

# =============================================================================
# Setup
# =============================================================================
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/game_${WORKER_ID}_${TIMESTAMP}.log"

echo "=============================================="
echo "  BGAI Game Worker (Distributed Mode)"
echo "=============================================="
echo "Platform:    $PLATFORM"
echo "Worker ID:   $WORKER_ID"
echo "Batch size:  $BATCH_SIZE"
echo "MCTS sims:   $MCTS_SIMULATIONS"
echo "MCTS nodes:  $MCTS_MAX_NODES"
echo "Head node:   $HEAD_IP:$RAY_PORT"
echo "Log file:    $LOG_FILE"
echo ""

# Join the Ray cluster first
join_ray_cluster || exit 1
echo ""

# =============================================================================
# Start game worker with auto-restart
# =============================================================================
STOP_FILE="$LOG_DIR/game_${WORKER_ID}.stop"
PID_FILE="$LOG_DIR/game_${WORKER_ID}.pid"

# Remove any old stop file
rm -f "$STOP_FILE"

echo "Starting game worker with auto-restart..."
echo "Stop with: touch $STOP_FILE"
echo ""

# Function to run worker with auto-restart
run_worker_loop() {
    RESTART_DELAY=5
    while true; do
        # Check for stop file
        if [[ -f "$STOP_FILE" ]]; then
            echo "$(date): Stop file detected. Exiting."
            rm -f "$STOP_FILE" "$PID_FILE"
            exit 0
        fi

        echo "$(date): Starting game worker..."
        python -m distributed.cli.main game-worker \
            --coordinator-address "auto" \
            --worker-id "$WORKER_ID" \
            --batch-size "$BATCH_SIZE" \
            --mcts-simulations "$MCTS_SIMULATIONS" \
            --mcts-max-nodes "$MCTS_MAX_NODES" \
            --temperature 1.0 \
            --redis-host "$REDIS_HOST" \
            --redis-port "$REDIS_PORT" \
            --redis-password "$REDIS_PASSWORD"

        EXIT_CODE=$?
        echo "$(date): Worker exited with code $EXIT_CODE"

        # Check for stop file again before restarting
        if [[ -f "$STOP_FILE" ]]; then
            echo "$(date): Stop file detected. Exiting."
            rm -f "$STOP_FILE" "$PID_FILE"
            exit 0
        fi

        echo "$(date): Restarting in $RESTART_DELAY seconds..."
        sleep $RESTART_DELAY
        # Exponential backoff up to 60 seconds
        RESTART_DELAY=$((RESTART_DELAY * 2))
        if [[ $RESTART_DELAY -gt 60 ]]; then
            RESTART_DELAY=60
        fi
    done
}

# Run in background
run_worker_loop >> "$LOG_FILE" 2>&1 &
WORKER_PID=$!
echo "$WORKER_PID" > "$PID_FILE"

echo "Game worker started with PID: $WORKER_PID"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Stop with:"
echo "  touch $STOP_FILE"
echo "  # or: kill $WORKER_PID"
