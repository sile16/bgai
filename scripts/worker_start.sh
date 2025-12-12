#!/bin/bash
# Unified worker script for BGAI distributed training
# Workers connect to Redis directly (no Ray required)
#
# Usage:
#   ./scripts/worker_start.sh                          # Start game worker (default)
#   ./scripts/worker_start.sh game                     # Start game worker
#   ./scripts/worker_start.sh eval                     # Start eval worker
#   ./scripts/worker_start.sh game eval                # Start both workers
#   ./scripts/worker_start.sh game --game-batch-size 32    # Override game batch size
#   ./scripts/worker_start.sh eval --eval-batch-size 16    # Override eval batch size
#   ./scripts/worker_start.sh game eval -g 32 -e 16        # Both with overrides
#
# Environment:
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
TAILSCALE_IP=$(grep -A5 "^head:" "$CONFIG_FILE" | grep "host:" | head -1 | sed 's/.*: *"\([^"]*\)".*/\1/')
LOCAL_IP=$(grep -A5 "^head:" "$CONFIG_FILE" | grep "host_local:" | sed 's/.*: *"\([^"]*\)".*/\1/')
REDIS_PORT=$(grep -A5 "^redis:" "$CONFIG_FILE" | grep "port:" | sed 's/.*: *\([0-9]*\).*/\1/')

# Fallback defaults
TAILSCALE_IP="${TAILSCALE_IP:-100.105.50.111}"
LOCAL_IP="${LOCAL_IP:-192.168.20.40}"
REDIS_PORT="${REDIS_PORT:-6379}"

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
        DEVICE_TYPE="cpu"
        # Force CPU on Mac - Metal/MPS has serialization issues with JAX
        export JAX_PLATFORMS=cpu
        echo "Detected macOS - forcing JAX to use CPU (JAX_PLATFORMS=cpu)"
    elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        PLATFORM="cuda"
        DEVICE_TYPE="cuda"
        # Set memory fraction based on actual JAX memory usage analysis
        # Game worker: 2.51 GB peak → 5 GB (0.21), Eval: ~1 GB → 2 GB (0.09)
        # This will be overridden per-worker if needed
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.21
        echo "JAX memory fraction: 0.21 (5 GB for game worker)"
    else
        PLATFORM="cpu"
        DEVICE_TYPE="cpu"
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

# Local tag for logs/stop files
WORKER_TAG_BASE="${HOSTNAME_SHORT}"

# =============================================================================
# Display configuration
# =============================================================================
echo "=============================================="
echo "  BGAI Worker"
echo "=============================================="
echo "Platform:     $PLATFORM ($DEVICE_TYPE)"
echo "Worker types: ${WORKER_TYPES[*]}"
echo "Worker tag:   $WORKER_TAG_BASE"
echo "Head node:    $HEAD_IP (Redis port: $REDIS_PORT)"
echo "MLflow:       http://$HEAD_IP:5000"
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
    local worker_tag="${WORKER_TAG_BASE}-game"
    local log_file="$LOG_DIR/game_${worker_tag}_${TIMESTAMP}.log"
    local stop_file="$LOG_DIR/game_${worker_tag}.stop"
    local pid_file="$LOG_DIR/game_${worker_tag}.pid"

    rm -f "$stop_file"

    echo "Starting game worker: $worker_tag"
    echo "  Log: $log_file"

    # Build command
    local cmd="python -m distributed.cli.main game-worker"
    cmd="$cmd --config-file $CONFIG_FILE"
    cmd="$cmd --head-ip $HEAD_IP"

    if [[ -n "$CUSTOM_GAME_BATCH" ]]; then
        cmd="$cmd --batch-size $CUSTOM_GAME_BATCH"
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
    local worker_tag="${WORKER_TAG_BASE}-eval"
    local log_file="$LOG_DIR/eval_${worker_tag}_${TIMESTAMP}.log"
    local stop_file="$LOG_DIR/eval_${worker_tag}.stop"
    local pid_file="$LOG_DIR/eval_${worker_tag}.pid"
    local eval_types=$(get_eval_types)

    rm -f "$stop_file"

    echo "Starting eval worker: $worker_tag"
    echo "  Eval types: $eval_types"
    echo "  Log: $log_file"

    # Build command
    local cmd="python -m distributed.cli.main eval-worker"
    cmd="$cmd --config-file $CONFIG_FILE"
    cmd="$cmd --head-ip $HEAD_IP"

    if [[ -n "$CUSTOM_EVAL_BATCH" ]]; then
        cmd="$cmd --batch-size $CUSTOM_EVAL_BATCH"
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
echo "  tail -f $LOG_DIR/*_${WORKER_TAG_BASE}*_${TIMESTAMP}.log"
echo ""
echo "Stop all workers:"
for worker_type in "${WORKER_TYPES[@]}"; do
    echo "  touch $LOG_DIR/${worker_type}_${WORKER_TAG_BASE}-${worker_type}.stop"
done
echo ""
echo "Check cluster status:"
echo "  ./scripts/status.sh"
