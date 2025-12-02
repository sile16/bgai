#!/bin/bash
# Start all head node services (coordinator, training worker, game worker)
# Configuration is loaded from configs/distributed.yaml
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
# Load configuration from YAML file
# =============================================================================
CONFIG_FILE="$PROJECT_DIR/configs/distributed.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi
echo "Loading config from: $CONFIG_FILE"

# Parse values from config (simple grep/sed approach)
TAILSCALE_IP=$(grep -A5 "^head:" "$CONFIG_FILE" | grep "host:" | head -1 | sed 's/.*: *"\([^"]*\)".*/\1/')
LOCAL_IP=$(grep -A5 "^head:" "$CONFIG_FILE" | grep "host_local:" | sed 's/.*: *"\([^"]*\)".*/\1/')
REDIS_PORT=$(grep -A5 "^redis:" "$CONFIG_FILE" | grep "port:" | sed 's/.*: *\([0-9]*\).*/\1/')
REDIS_PASSWORD=$(grep -A5 "^redis:" "$CONFIG_FILE" | grep "password:" | sed 's/.*: *"\([^"]*\)".*/\1/')

# Fallback defaults
TAILSCALE_IP="${TAILSCALE_IP:-100.105.50.111}"
LOCAL_IP="${LOCAL_IP:-192.168.20.40}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-bgai-password}"

# Try Tailscale IP first, fallback to local IP
if ping -c 1 -W 1 "$TAILSCALE_IP" &>/dev/null; then
    HEAD_IP="$TAILSCALE_IP"
    echo "Using Tailscale IP: $HEAD_IP"
else
    HEAD_IP="$LOCAL_IP"
    echo "Tailscale unavailable, using local IP: $HEAD_IP"
fi
REDIS_HOST="$HEAD_IP"

CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"
LOG_DIR="$PROJECT_DIR/logs"

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
echo "Config file: $CONFIG_FILE"
echo "Project dir: $PROJECT_DIR"
echo "Logs dir:    $LOG_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
echo ""

# =============================================================================
# Stop existing processes
# =============================================================================
echo "[1/5] Stopping any existing workers..."
pkill -f "distributed.cli.main" 2>/dev/null || true
sleep 2

# =============================================================================
# Set PYTHONPATH
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
# Start Prometheus metrics with custom BGAI config
# =============================================================================
echo "[2/5] Starting Prometheus metrics..."
PROMETHEUS_BIN="$PROJECT_DIR/tools/prometheus/prometheus"
PROMETHEUS_CONFIG="$PROJECT_DIR/tools/prometheus_bgai.yml"

# Find prometheus binary
if [[ ! -f "$PROMETHEUS_BIN" ]]; then
    PROMETHEUS_BIN=$(find "$PROJECT_DIR" -name "prometheus" -type f -executable 2>/dev/null | head -1)
fi

if [[ -f "$PROMETHEUS_BIN" ]]; then
    "$PROMETHEUS_BIN" \
        --config.file "$PROMETHEUS_CONFIG" \
        --web.enable-lifecycle \
        > "$LOG_DIR/prometheus_$TIMESTAMP.log" 2>&1 &
    PROMETHEUS_PID=$!
    echo "       Prometheus PID: $PROMETHEUS_PID"
else
    echo "       WARNING: Prometheus not found. Metrics collection disabled."
    PROMETHEUS_PID=""
fi
if [[ -n "$PROMETHEUS_PID" ]]; then
    echo "       Prometheus UI: http://$HEAD_IP:9090"
fi
sleep 2

# =============================================================================
# Start Grafana for metrics visualization
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
import time
updater = start_discovery_updater(
    redis_host='$REDIS_HOST',
    redis_port=$REDIS_PORT,
    redis_password='$REDIS_PASSWORD',
    update_interval=15,
)
print('Discovery updater started')
while True:
    time.sleep(3600)
" > "$LOG_DIR/discovery_updater_$TIMESTAMP.log" 2>&1 &
DISCOVERY_PID=$!
echo "       Discovery updater PID: $DISCOVERY_PID"
sleep 2

# =============================================================================
# Start Coordinator (uses config file for all settings)
# =============================================================================
echo "[3/4] Starting coordinator..."
python -m distributed.cli.main coordinator \
    --config-file "$CONFIG_FILE" \
    --dashboard \
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
# Start Training Worker (uses config file, 0.5 GPU to share with game worker)
# =============================================================================
echo "[4/4] Starting training worker..."
python -m distributed.cli.main training-worker \
    --config-file "$CONFIG_FILE" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --num-gpus 0.5 \
    > "$LOG_DIR/training_$TIMESTAMP.log" 2>&1 &
TRAIN_PID=$!
echo "       Training worker PID: $TRAIN_PID"
echo "       Log: $LOG_DIR/training_$TIMESTAMP.log"

# =============================================================================
# Start GPU Game Worker (uses config file, 0.5 GPU to share with training worker)
# =============================================================================
echo "       Starting GPU game worker..."
python -m distributed.cli.main game-worker \
    --config-file "$CONFIG_FILE" \
    --worker-id "gpu-head" \
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
echo "  Grafana:         http://$HEAD_IP:3000"
echo "  Prometheus:      http://$HEAD_IP:9090"
echo ""
echo "Logs:"
if [[ -n "$PROMETHEUS_PID" ]]; then
echo "  tail -f $LOG_DIR/prometheus_$TIMESTAMP.log"
fi
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
echo "Remote workers connect via Redis: $REDIS_HOST:$REDIS_PORT"
