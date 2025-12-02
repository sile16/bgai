#!/bin/bash
# Unified worker script for BGAI distributed training
# Handles Ray connection, worker type detection, and hardware-specific settings
#
# Usage:
#   ./scripts/start_worker.sh                          # Start game worker (default)
#   ./scripts/start_worker.sh game                     # Start game worker
#   ./scripts/start_worker.sh eval                     # Start eval worker
#   ./scripts/start_worker.sh game eval                # Start both workers
#   ./scripts/start_worker.sh game --game-batch-size 32    # Override game batch size
#   ./scripts/start_worker.sh eval --eval-batch-size 16    # Override eval batch size
#   ./scripts/start_worker.sh game eval -g 32 -e 16        # Both with overrides
#
# Environment:
#   WORKER_ID       - Override auto-generated worker ID
#   GAME_BATCH_SIZE - Override game batch size
#   EVAL_BATCH_SIZE - Override eval batch size

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
fi
echo "Using Python: $(which python)"

# =============================================================================
# Load configuration from YAML file
# =============================================================================
CONFIG_FILE="$PROJECT_DIR/configs/distributed.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Parse head node IPs from config
TAILSCALE_IP=$(grep "head_ip:" "$CONFIG_FILE" | head -1 | sed 's/.*: *"\([^"]*\)".*/\1/')
LOCAL_IP=$(grep "head_ip_local:" "$CONFIG_FILE" | sed 's/.*: *"\([^"]*\)".*/\1/')
RAY_PORT=$(grep "gcs_port:" "$CONFIG_FILE" | sed 's/.*: *\([0-9]*\).*/\1/')

# Fallback defaults
TAILSCALE_IP="${TAILSCALE_IP:-100.105.50.111}"
LOCAL_IP="${LOCAL_IP:-192.168.20.40}"
RAY_PORT="${RAY_PORT:-6380}"

# Try Tailscale IP first, fallback to local IP
if ping -c 1 -W 1 "$TAILSCALE_IP" &>/dev/null; then
    HEAD_IP="$TAILSCALE_IP"
else
    HEAD_IP="$LOCAL_IP"
fi

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# =============================================================================
# Detect platform and capabilities
# =============================================================================
detect_platform() {
    OS_TYPE=$(uname -s)

    if [[ "$OS_TYPE" == "Darwin" ]]; then
        PLATFORM="mac"
        DEVICE_TYPE="metal"
        # Mac: eval runs on CPU to free up Metal for game workers
        EVAL_NUM_GPUS=0
        GAME_NUM_GPUS=0
    elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        PLATFORM="cuda"
        DEVICE_TYPE="cuda"
        # CUDA: can run eval on GPU
        EVAL_NUM_GPUS=0.25
        GAME_NUM_GPUS=0.5
    else
        PLATFORM="cpu"
        DEVICE_TYPE="cpu"
        EVAL_NUM_GPUS=0
        GAME_NUM_GPUS=0
    fi

    # Check for gnubg
    if command -v gnubg &>/dev/null; then
        GNUBG_AVAILABLE=true
    else
        GNUBG_AVAILABLE=false
    fi

    HOSTNAME_SHORT=$(hostname -s 2>/dev/null || hostname)
}

detect_platform

# =============================================================================
# Parse arguments
# =============================================================================
WORKER_TYPES=()
EXTRA_ARGS=""
CUSTOM_WORKER_ID="${WORKER_ID:-}"
CUSTOM_GAME_BATCH="${GAME_BATCH_SIZE:-}"
CUSTOM_EVAL_BATCH="${EVAL_BATCH_SIZE:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        game|eval)
            WORKER_TYPES+=("$1")
            shift
            ;;
        -g|--game-batch-size)
            CUSTOM_GAME_BATCH="$2"
            shift 2
            ;;
        -e|--eval-batch-size)
            CUSTOM_EVAL_BATCH="$2"
            shift 2
            ;;
        --worker-id)
            CUSTOM_WORKER_ID="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [game] [eval] [options]"
            echo ""
            echo "Worker types:"
            echo "  game    Start game generation worker (default)"
            echo "  eval    Start evaluation worker"
            echo ""
            echo "Options:"
            echo "  -g, --game-batch-size N   Override game batch size"
            echo "  -e, --eval-batch-size N   Override eval batch size"
            echo "  --worker-id ID            Override worker ID"
            echo "  -h, --help                Show this help"
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Default to game worker if none specified
if [[ ${#WORKER_TYPES[@]} -eq 0 ]]; then
    WORKER_TYPES=("game")
fi

# Generate worker ID if not provided
if [[ -z "$CUSTOM_WORKER_ID" ]]; then
    WORKER_ID_BASE="${PLATFORM}-${HOSTNAME_SHORT}"
else
    WORKER_ID_BASE="$CUSTOM_WORKER_ID"
fi

# =============================================================================
# Display configuration
# =============================================================================
echo "=============================================="
echo "  BGAI Worker"
echo "=============================================="
echo "Platform:     $PLATFORM ($DEVICE_TYPE)"
echo "Worker types: ${WORKER_TYPES[*]}"
echo "Worker ID:    $WORKER_ID_BASE"
echo "Head node:    $HEAD_IP:$RAY_PORT"
echo "Config file:  $CONFIG_FILE"
if [[ -n "$CUSTOM_GAME_BATCH" ]]; then
    echo "Game batch:   $CUSTOM_GAME_BATCH (override)"
fi
if [[ -n "$CUSTOM_EVAL_BATCH" ]]; then
    echo "Eval batch:   $CUSTOM_EVAL_BATCH (override)"
fi
echo "gnubg:        $(if $GNUBG_AVAILABLE; then echo "available"; else echo "not found"; fi)"
echo ""

# =============================================================================
# Connect to Ray cluster
# =============================================================================
connect_ray_cluster() {
    echo "Checking Ray connection..."

    # Check if already connected to the correct cluster
    if ray status &>/dev/null; then
        CURRENT_HEAD=$(ray status 2>/dev/null | grep -oP 'address.*:\K[0-9.]+' | head -1 || echo "")
        if [[ "$CURRENT_HEAD" == "$HEAD_IP" ]] || [[ "$CURRENT_HEAD" == "0.0.0.0" ]]; then
            echo "Already connected to Ray cluster"
            return 0
        else
            echo "Connected to different cluster, reconnecting..."
            ray stop --force 2>/dev/null || true
            sleep 2
        fi
    fi

    # Enable multi-node clusters on Mac/Windows
    export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1

    echo "Joining Ray cluster at $HEAD_IP:$RAY_PORT..."

    # Get system info for node labeling
    local node_name="${HOSTNAME_SHORT}"
    local num_cpus=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "4")
    local mem_bytes=$(free -b 2>/dev/null | awk '/^Mem:/{print $2}' || sysctl -n hw.memsize 2>/dev/null || echo "8000000000")

    # Start Ray and join cluster with node info
    if ! ray start \
        --address="$HEAD_IP:$RAY_PORT" \
        --node-name="$node_name" \
        --num-cpus="$num_cpus" \
        --resources="{\"${PLATFORM}\": 1}" \
        2>&1; then
        echo "ERROR: Failed to join Ray cluster"
        echo "Make sure the head node is running: ./scripts/start_all_head.sh"
        return 1
    fi

    # Wait for connection to stabilize
    sleep 3

    # Verify connection
    if ray status &>/dev/null; then
        NODE_COUNT=$(ray status 2>/dev/null | grep -c "node_" || echo "0")
        echo "Successfully joined Ray cluster ($NODE_COUNT nodes)"
        return 0
    else
        echo "ERROR: Ray connection verification failed"
        return 1
    fi
}

connect_ray_cluster || exit 1
echo ""

# =============================================================================
# Build eval types based on platform
# =============================================================================
get_eval_types() {
    local types="random,self_play"

    if $GNUBG_AVAILABLE; then
        types="gnubg,$types"
    fi

    echo "$types"
}

# =============================================================================
# Start workers
# =============================================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIDS=()

start_game_worker() {
    local worker_id="${WORKER_ID_BASE}-game"
    local log_file="$LOG_DIR/game_${worker_id}_${TIMESTAMP}.log"
    local stop_file="$LOG_DIR/game_${worker_id}.stop"
    local pid_file="$LOG_DIR/game_${worker_id}.pid"

    rm -f "$stop_file"

    echo "Starting game worker: $worker_id"
    echo "  Log: $log_file"

    # Build command
    local cmd="python -m distributed.cli.main game-worker"
    cmd="$cmd --config-file $CONFIG_FILE"
    cmd="$cmd --worker-id $worker_id"

    if [[ -n "$CUSTOM_GAME_BATCH" ]]; then
        cmd="$cmd --batch-size $CUSTOM_GAME_BATCH"
    fi

    if [[ "$GAME_NUM_GPUS" != "0" ]]; then
        cmd="$cmd --num-gpus $GAME_NUM_GPUS"
    fi

    # Run with auto-restart loop
    (
        RESTART_DELAY=5
        while true; do
            if [[ -f "$stop_file" ]]; then
                echo "$(date): Stop file detected. Exiting."
                rm -f "$stop_file" "$pid_file"
                exit 0
            fi

            echo "$(date): Starting game worker..."
            $cmd $EXTRA_ARGS

            if [[ -f "$stop_file" ]]; then
                rm -f "$stop_file" "$pid_file"
                exit 0
            fi

            echo "$(date): Worker exited, restarting in ${RESTART_DELAY}s..."
            sleep $RESTART_DELAY
            RESTART_DELAY=$((RESTART_DELAY * 2))
            [[ $RESTART_DELAY -gt 60 ]] && RESTART_DELAY=60
        done
    ) >> "$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"
    PIDS+=($pid)
    echo "  PID: $pid"
    echo "  Stop: touch $stop_file"
}

start_eval_worker() {
    local worker_id="${WORKER_ID_BASE}-eval"
    local log_file="$LOG_DIR/eval_${worker_id}_${TIMESTAMP}.log"
    local stop_file="$LOG_DIR/eval_${worker_id}.stop"
    local pid_file="$LOG_DIR/eval_${worker_id}.pid"
    local eval_types=$(get_eval_types)

    rm -f "$stop_file"

    echo "Starting eval worker: $worker_id"
    echo "  Eval types: $eval_types"
    echo "  Log: $log_file"

    # Build command
    local cmd="python -m distributed.cli.main eval-worker"
    cmd="$cmd --config-file $CONFIG_FILE"
    cmd="$cmd --worker-id $worker_id"
    cmd="$cmd --eval-types $eval_types"

    if [[ -n "$CUSTOM_EVAL_BATCH" ]]; then
        cmd="$cmd --batch-size $CUSTOM_EVAL_BATCH"
    fi

    if [[ "$EVAL_NUM_GPUS" != "0" ]]; then
        cmd="$cmd --num-gpus $EVAL_NUM_GPUS"
    fi

    # Run with auto-restart loop
    (
        RESTART_DELAY=5
        while true; do
            if [[ -f "$stop_file" ]]; then
                echo "$(date): Stop file detected. Exiting."
                rm -f "$stop_file" "$pid_file"
                exit 0
            fi

            echo "$(date): Starting eval worker..."
            $cmd $EXTRA_ARGS

            if [[ -f "$stop_file" ]]; then
                rm -f "$stop_file" "$pid_file"
                exit 0
            fi

            echo "$(date): Worker exited, restarting in ${RESTART_DELAY}s..."
            sleep $RESTART_DELAY
            RESTART_DELAY=$((RESTART_DELAY * 2))
            [[ $RESTART_DELAY -gt 60 ]] && RESTART_DELAY=60
        done
    ) >> "$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"
    PIDS+=($pid)
    echo "  PID: $pid"
    echo "  Stop: touch $stop_file"
}

# Start requested worker types
for worker_type in "${WORKER_TYPES[@]}"; do
    case "$worker_type" in
        game)
            start_game_worker
            ;;
        eval)
            start_eval_worker
            ;;
    esac
    echo ""
done

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "  Workers Started"
echo "=============================================="
echo ""
echo "Monitor logs:"
echo "  tail -f $LOG_DIR/*_${WORKER_ID_BASE}*_${TIMESTAMP}.log"
echo ""
echo "Stop all workers:"
for worker_type in "${WORKER_TYPES[@]}"; do
    echo "  touch $LOG_DIR/${worker_type}_${WORKER_ID_BASE}-${worker_type}.stop"
done
echo ""
echo "Check cluster status:"
echo "  ray status"
