#!/bin/bash
# Start a game worker with auto-detection of platform and configuration
# Configuration is loaded from configs/distributed.yaml
#
# Usage:
#   ./scripts/start_game_worker.sh              # Auto-detect everything
#   ./scripts/start_game_worker.sh my-worker    # Custom worker ID
#   ./scripts/start_game_worker.sh --batch-size 32  # Override batch size

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
# Configuration from YAML file
# =============================================================================
CONFIG_FILE="$PROJECT_DIR/configs/distributed.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Parse head node IPs from config (simple grep/sed approach)
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
    echo "Using Tailscale IP: $HEAD_IP"
else
    HEAD_IP="$LOCAL_IP"
    echo "Tailscale unavailable, using local IP: $HEAD_IP"
fi

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# =============================================================================
# Parse arguments
# =============================================================================
WORKER_ID=""
EXTRA_ARGS=""

# Parse worker ID (first positional arg that doesn't start with --)
for arg in "$@"; do
    if [[ "$arg" != --* ]] && [[ -z "$WORKER_ID" ]]; then
        WORKER_ID="$arg"
    else
        EXTRA_ARGS="$EXTRA_ARGS $arg"
    fi
done

# Generate worker ID if not provided
if [[ -z "$WORKER_ID" ]]; then
    HOSTNAME_SHORT=$(hostname -s 2>/dev/null || hostname)
    # Detect platform for worker ID prefix
    OS_TYPE=$(uname -s)
    if [[ "$OS_TYPE" == "Darwin" ]]; then
        PLATFORM_TAG="mac"
    elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        PLATFORM_TAG="cuda"
    else
        PLATFORM_TAG="cpu"
    fi
    WORKER_ID="${PLATFORM_TAG}-${HOSTNAME_SHORT}"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/game_${WORKER_ID}_${TIMESTAMP}.log"
STOP_FILE="$LOG_DIR/game_${WORKER_ID}.stop"
PID_FILE="$LOG_DIR/game_${WORKER_ID}.pid"

# =============================================================================
# Display configuration
# =============================================================================
echo "=============================================="
echo "  BGAI Game Worker"
echo "=============================================="
echo "Worker ID:   $WORKER_ID"
echo "Config file: $CONFIG_FILE"
echo "Redis:       $HEAD_IP:$REDIS_PORT"
echo "Log file:    $LOG_FILE"
echo "Extra args:  $EXTRA_ARGS"
echo ""

# =============================================================================
# Start game worker with auto-restart
# =============================================================================
rm -f "$STOP_FILE"

echo "Starting game worker with auto-restart..."
echo "Stop with: touch $STOP_FILE"
echo ""

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
            --config-file "$CONFIG_FILE" \
            --worker-id "$WORKER_ID" \
            $EXTRA_ARGS

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
