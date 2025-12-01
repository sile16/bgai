#!/bin/bash
# Start an evaluation worker with auto-detection of platform (Mac/Linux+CUDA/TPU)
# Runs in background with logs to file
#
# The eval worker periodically evaluates the current model against baselines
# (e.g., GNUBG, previous checkpoints, random policy)
#
# Usage: ./scripts/start_eval_worker.sh [worker_id]

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
RAY_CLIENT_PORT="10001"
LOG_DIR="$PROJECT_DIR/logs"

# Evaluation settings
EVAL_GAMES=100          # Games per evaluation round
EVAL_INTERVAL=300       # Seconds between evaluations
EVAL_TYPES="${EVAL_TYPES:-gnubg,random,self_play}"  # Comma-separated eval types

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
        BATCH_SIZE=32
        MCTS_SIMULATIONS=200
        MCTS_MAX_NODES=800
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        PLATFORM_TAG="cuda"
        ;;
    mac)
        BATCH_SIZE=8
        MCTS_SIMULATIONS=100
        MCTS_MAX_NODES=400
        export JAX_PLATFORMS="cpu"
        PLATFORM_TAG="mac"
        ;;
    tpu)
        # Future TPU support
        BATCH_SIZE=64
        MCTS_SIMULATIONS=200
        MCTS_MAX_NODES=800
        PLATFORM_TAG="tpu"
        ;;
    cpu|*)
        BATCH_SIZE=4
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
    WORKER_ID="eval-${PLATFORM_TAG}-${HOSTNAME_SHORT}"
fi

# =============================================================================
# Setup
# =============================================================================
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/eval_${WORKER_ID}_${TIMESTAMP}.log"

echo "=============================================="
echo "  BGAI Evaluation Worker"
echo "=============================================="
echo "Platform:      $PLATFORM"
echo "Worker ID:     $WORKER_ID"
echo "Batch size:    $BATCH_SIZE"
echo "MCTS sims:     $MCTS_SIMULATIONS"
echo "Eval games:    $EVAL_GAMES"
echo "Eval interval: ${EVAL_INTERVAL}s"
echo "Eval types:    $EVAL_TYPES"
echo "Head node:     $HEAD_IP:$RAY_CLIENT_PORT"
echo "Log file:      $LOG_FILE"
echo ""

# =============================================================================
# Start eval worker
# Note: eval-worker command needs to be implemented in CLI
# For now, this is a placeholder that shows the intended interface
# =============================================================================
echo "Starting evaluation worker..."

# Check if eval-worker command exists
if python -m distributed.cli.main --help 2>&1 | grep -q "eval-worker"; then
    python -m distributed.cli.main eval-worker \
        --coordinator-address "ray://$HEAD_IP:$RAY_CLIENT_PORT" \
        --worker-id "$WORKER_ID" \
        --batch-size "$BATCH_SIZE" \
        --mcts-simulations "$MCTS_SIMULATIONS" \
        --mcts-max-nodes "$MCTS_MAX_NODES" \
        --eval-games "$EVAL_GAMES" \
        --eval-interval "$EVAL_INTERVAL" \
        --eval-types "$EVAL_TYPES" \
        --redis-host "$REDIS_HOST" \
        --redis-port "$REDIS_PORT" \
        --redis-password "$REDIS_PASSWORD" \
        > "$LOG_FILE" 2>&1 &

    WORKER_PID=$!
    echo "Evaluation worker started with PID: $WORKER_PID"
    echo "$WORKER_PID" > "$LOG_DIR/eval_${WORKER_ID}.pid"
else
    echo "WARNING: eval-worker command not yet implemented in CLI."
    echo ""
    echo "The evaluation worker will:"
    echo "  1. Periodically fetch current model weights"
    echo "  2. Play $EVAL_GAMES games against baseline (GNUBG/random)"
    echo "  3. Report win rate to coordinator"
    echo ""
    echo "To implement, add 'eval-worker' command to distributed/cli/main.py"
    exit 1
fi

echo ""
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Stop with:"
echo "  kill $WORKER_PID"
