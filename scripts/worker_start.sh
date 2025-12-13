#!/bin/bash
# Unified worker script for BGAI distributed training
# Workers connect to Redis directly (no Ray required)
#
# Usage:
#   ./scripts/worker_start.sh                          # Start game worker (default)
#   ./scripts/worker_start.sh game                     # Start game worker
#   ./scripts/worker_start.sh eval                     # Start eval worker
#   ./scripts/worker_start.sh game eval                # Start both workers
#   ./scripts/worker_start.sh --cpu game               # Force CPU worker (on GPU machine)
#   ./scripts/worker_start.sh --gpu game               # Force GPU worker (if available)
#   ./scripts/worker_start.sh --tpu game               # Force TPU worker (if available)
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
FORCE_DEVICE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        game|eval)
            WORKER_TYPES+=("$1")
            shift
            ;;
        --cpu)
            FORCE_DEVICE="cpu"
            shift
            ;;
        --gpu)
            FORCE_DEVICE="gpu"
            shift
            ;;
        --tpu)
            FORCE_DEVICE="tpu"
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
            echo "      --cpu                Force CPU worker (JAX_PLATFORMS=cpu)"
            echo "      --gpu                Force GPU worker (JAX_PLATFORMS=cuda)"
            echo "      --tpu                Force TPU worker (JAX_PLATFORMS=tpu)"
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

apply_force_device() {
    if [[ -z "$FORCE_DEVICE" ]]; then
        return 0
    fi

    if [[ "$OS_TYPE" == "Darwin" && "$FORCE_DEVICE" != "cpu" ]]; then
        echo "ERROR: --$FORCE_DEVICE requested on macOS; only --cpu is supported"
        exit 1
    fi

    case "$FORCE_DEVICE" in
        cpu)
            PLATFORM="cpu"
            DEVICE_TYPE="cpu"
            export JAX_PLATFORMS=cpu
            unset XLA_PYTHON_CLIENT_MEM_FRACTION 2>/dev/null || true
            ;;
        gpu)
            if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null; then
                echo "ERROR: --gpu requested but no working NVIDIA GPU detected (nvidia-smi failed)"
                exit 1
            fi
            PLATFORM="cuda"
            DEVICE_TYPE="cuda"
            export JAX_PLATFORMS=cuda
            export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.21}"
            ;;
        tpu)
            PLATFORM="tpu"
            DEVICE_TYPE="tpu"
            export JAX_PLATFORMS=tpu
            unset XLA_PYTHON_CLIENT_MEM_FRACTION 2>/dev/null || true
            ;;
        *)
            echo "ERROR: Unknown device override: $FORCE_DEVICE"
            exit 1
            ;;
    esac
}

apply_force_device

# Default to game worker if none specified
if [[ ${#WORKER_TYPES[@]} -eq 0 ]]; then
    WORKER_TYPES=("game")
fi

# Local tag for logs/stop files (include device so CPU+GPU can coexist)
WORKER_TAG_BASE="${HOSTNAME_SHORT}-${DEVICE_TYPE}"

# =============================================================================
# Process helpers (one worker type per device)
# =============================================================================
parse_arg_value() {
    local arg_name="$1"
    shift
    local prev=""
    for token in "$@"; do
        if [[ "$prev" == "$arg_name" ]]; then
            echo "$token"
            return 0
        fi
        case "$token" in
            "$arg_name="*)
                echo "${token#*=}"
                return 0
                ;;
        esac
        prev="$token"
    done
    return 1
}

describe_worker_process() {
    local pid="$1"
    local worker_kind="$2" # game|eval
    local cmd rss_kb rss_mb gpu_mem_mib batch_size

    cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    if [[ -z "$cmd" ]]; then
        echo "PID $pid (exited)"
        return 0
    fi

    # Extract batch size if present in CLI.
    read -r -a tokens <<<"$cmd"
    batch_size="$(parse_arg_value --batch-size "${tokens[@]}" || true)"
    batch_size="${batch_size:-<from config>}"

    rss_kb="$(ps -p "$pid" -o rss= 2>/dev/null | tr -d ' ' || true)"
    if [[ -n "$rss_kb" ]]; then
        rss_mb=$((rss_kb / 1024))
    else
        rss_mb="?"
    fi

    gpu_mem_mib=""
    if command -v nvidia-smi &>/dev/null; then
        gpu_mem_mib="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' -v p="$pid" '$1+0==p {gsub(/ /,"",$2); print $2; exit}')"
    fi
    if [[ -n "$gpu_mem_mib" ]]; then
        echo "Current ${WORKER_TAG_BASE}-${worker_kind}: batch_size=${batch_size}, rss=${rss_mb}MB, gpu_mem=${gpu_mem_mib}MiB"
    else
        echo "Current ${WORKER_TAG_BASE}-${worker_kind}: batch_size=${batch_size}, rss=${rss_mb}MB"
    fi
}

find_existing_worker_pid() {
    local pid_file="$1"
    local worker_subcommand="$2" # game-worker|eval-worker

    if [[ -f "$pid_file" ]]; then
        local pid
        pid="$(cat "$pid_file" 2>/dev/null || true)"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
    fi

    # Best-effort fallback: find a matching worker started with this script
    # (it includes both --config-file and --head-ip flags).
    if command -v pgrep &>/dev/null; then
        local pid
        pid="$(pgrep -f "python -m distributed\\.cli\\.main ${worker_subcommand}.*--config-file ${CONFIG_FILE}.*--head-ip ${HEAD_IP}" | head -n 1 || true)"
        if [[ -n "$pid" ]]; then
            echo "$pid"
            return 0
        fi
    fi

    return 1
}

stop_existing_worker() {
    local pid="$1"
    local stop_file="$2"
    local pid_file="$3"

    touch "$stop_file"

    local deadline=$((SECONDS + 20))
    while kill -0 "$pid" 2>/dev/null; do
        if [[ $SECONDS -ge $deadline ]]; then
            break
        fi
        sleep 0.5
    done

    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        sleep 0.5
    fi
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
    fi

    rm -f "$stop_file" "$pid_file"
}

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

    local existing_pid=""
    existing_pid="$(find_existing_worker_pid "$pid_file" "game-worker" || true)"
    if [[ -n "$existing_pid" ]]; then
        describe_worker_process "$existing_pid" "game"
        echo "Restarting ${worker_tag} with new params..."
        stop_existing_worker "$existing_pid" "$stop_file" "$pid_file"
    fi

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

    local existing_pid=""
    existing_pid="$(find_existing_worker_pid "$pid_file" "eval-worker" || true)"
    if [[ -n "$existing_pid" ]]; then
        describe_worker_process "$existing_pid" "eval"
        echo "Restarting ${worker_tag} with new params..."
        stop_existing_worker "$existing_pid" "$stop_file" "$pid_file"
    fi

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
