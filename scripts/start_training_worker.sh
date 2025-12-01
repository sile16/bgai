#!/bin/bash
# Start a training worker with auto-detection of platform (Mac/Linux+CUDA/TPU)
# Runs in background with logs to file
#
# Note: Training is typically done on the head node with GPU, but this script
# allows running on other machines if needed.
#
# Usage: ./scripts/start_training_worker.sh [worker_id]

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
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"
LOG_DIR="$PROJECT_DIR/logs"

# Training hyperparameters
LEARNING_RATE="3e-4"
MIN_BUFFER_SIZE=1000
WEIGHT_PUSH_INTERVAL=10
CHECKPOINT_INTERVAL=1000

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
        BATCH_SIZE=256
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        PLATFORM_TAG="cuda"
        ;;
    mac)
        BATCH_SIZE=64
        export JAX_PLATFORMS="cpu"
        PLATFORM_TAG="mac"
        echo "WARNING: Training on Mac CPU will be slow. Consider using GPU server."
        ;;
    tpu)
        # Future TPU support
        BATCH_SIZE=512
        PLATFORM_TAG="tpu"
        ;;
    cpu|*)
        BATCH_SIZE=64
        export JAX_PLATFORMS="cpu"
        PLATFORM_TAG="cpu"
        echo "WARNING: Training on CPU will be slow. Consider using GPU server."
        ;;
esac

# =============================================================================
# Worker ID
# =============================================================================
if [[ -n "$1" ]]; then
    WORKER_ID="$1"
else
    HOSTNAME_SHORT=$(hostname -s 2>/dev/null || hostname)
    WORKER_ID="train-${PLATFORM_TAG}-${HOSTNAME_SHORT}"
fi

# =============================================================================
# Setup
# =============================================================================
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_${WORKER_ID}_${TIMESTAMP}.log"

echo "=============================================="
echo "  BGAI Training Worker"
echo "=============================================="
echo "Platform:      $PLATFORM"
echo "Worker ID:     $WORKER_ID"
echo "Batch size:    $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Head node:     $HEAD_IP:$RAY_CLIENT_PORT"
echo "Checkpoints:   $CHECKPOINT_DIR"
echo "Log file:      $LOG_FILE"
echo ""

# =============================================================================
# Start training worker
# =============================================================================
echo "Starting training worker..."
python -m distributed.cli.main training-worker \
    --coordinator-address "ray://$HEAD_IP:$RAY_CLIENT_PORT" \
    --worker-id "$WORKER_ID" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --min-buffer-size "$MIN_BUFFER_SIZE" \
    --weight-push-interval "$WEIGHT_PUSH_INTERVAL" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --redis-host "$REDIS_HOST" \
    --redis-port "$REDIS_PORT" \
    --redis-password "$REDIS_PASSWORD" \
    > "$LOG_FILE" 2>&1 &

WORKER_PID=$!
echo "Training worker started with PID: $WORKER_PID"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Stop with:"
echo "  kill $WORKER_PID"

# Save PID for later
echo "$WORKER_PID" > "$LOG_DIR/training_${WORKER_ID}.pid"
