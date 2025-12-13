#!/bin/bash
# Start all head node services:
#   - MLflow, Prometheus, Grafana
#   - Prometheus discovery updater
#   - Coordinator
#   - Training worker (GPU)
#   - Game worker (GPU)
#   - Eval worker (CPU)
#
# Usage:
#   ./scripts/head_start.sh            # restart/continue existing run
#   ./scripts/head_start.sh --new      # wipe run state + restart
#   ./scripts/head_start.sh --continue # same as default
#   ./scripts/head_start.sh --config-file path/to.yaml
#
# Env overrides:
#   TRAIN_MEM_GB, GAME_MEM_GB, EVAL_JAX_PLATFORMS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODE="continue"
CONFIG_FILE="$PROJECT_DIR/configs/distributed.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --new) MODE="new"; shift ;;
    --continue) MODE="continue"; shift ;;
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--new|--continue] [--config-file PATH]"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

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
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi
echo "Loading config from: $CONFIG_FILE"

TAILSCALE_IP=$(grep -A5 "^head:" "$CONFIG_FILE" | grep "host:" | head -1 | awk -F'"' '{print $2}')
LOCAL_IP=$(grep -A5 "^head:" "$CONFIG_FILE" | grep "host_local:" | awk -F'"' '{print $2}')
REDIS_PORT=$(grep -A20 "^redis:" "$CONFIG_FILE" | grep "port:" | head -1 | awk '{print $2}')
REDIS_PASSWORD=$(grep -A20 "^redis:" "$CONFIG_FILE" | grep "password:" | head -1 | awk -F'"' '{print $2}')

TAILSCALE_IP="${TAILSCALE_IP:-100.105.50.111}"
LOCAL_IP="${LOCAL_IP:-192.168.20.40}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-bgai-password}"

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
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "  BGAI Distributed Training - Head Node"
echo "=============================================="
echo "Mode:        $MODE"
echo "Head IP:     $HEAD_IP"
echo "Redis:       $REDIS_HOST:$REDIS_PORT"
echo "Config file: $CONFIG_FILE"
echo ""

# =============================================================================
# Stop existing processes
# =============================================================================
echo "[1/7] Stopping existing services..."
"$SCRIPT_DIR/stop_all.sh"
sleep 2

# =============================================================================
# NEW mode: clear run state
# =============================================================================
if [[ "$MODE" == "new" ]]; then
  echo "[2/7] Clearing run state (Redis, checkpoints, logs)..."
  if ! command -v redis-cli &>/dev/null; then
    echo "ERROR: redis-cli not found but --new was requested."
    exit 1
  fi

  redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning KEYS "bgai:*" | \
      xargs -r redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning DEL >/dev/null 2>&1 || true
  redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning KEYS "buffer:*" | \
      xargs -r redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning DEL >/dev/null 2>&1 || true

  rm -rf "$CHECKPOINT_DIR/"* 2>/dev/null || true
  rm -f "$LOG_DIR/"*.log 2>/dev/null || true

  RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
  redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning SET bgai:run:id "$RUN_ID" >/dev/null
  redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning SET bgai:run:status "running" >/dev/null
  redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning SET bgai:model:version "0" >/dev/null
  echo "New run ID: $RUN_ID"
fi

# =============================================================================
# Environment
# =============================================================================
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

EVAL_JAX_PLATFORMS="${EVAL_JAX_PLATFORMS:-cpu}"

# Prefer fixed GPU memory caps from config; can override via env.
TRAIN_MEM_GB="${TRAIN_MEM_GB:-$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['device_configs']['cuda']['train_gpu_memory_gb'])")}"
GAME_MEM_GB="${GAME_MEM_GB:-$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['device_configs']['cuda']['game_gpu_memory_gb'])")}"

# Optional per-head batch overrides.
HEAD_TRAIN_BATCH_SIZE="${HEAD_TRAIN_BATCH_SIZE:-}"
HEAD_GAME_BATCH_SIZE="${HEAD_GAME_BATCH_SIZE:-}"

# =============================================================================
# Start MLflow
# =============================================================================
echo "[3/7] Starting MLflow..."
MLFLOW_DIR="$PROJECT_DIR/mlruns"
mkdir -p "$MLFLOW_DIR"
python -m mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "sqlite:///$MLFLOW_DIR/mlflow.db" \
    --default-artifact-root "$MLFLOW_DIR/artifacts" \
    --allowed-hosts "*" \
    > "$LOG_DIR/mlflow_$TIMESTAMP.log" 2>&1 &
MLFLOW_PID=$!
echo "       MLflow PID: $MLFLOW_PID"
sleep 2

# =============================================================================
# Start Prometheus
# =============================================================================
echo "[4/7] Starting Prometheus..."
PROM_DIR="$PROJECT_DIR/tools/prometheus"
PROM_BIN="$PROM_DIR/prometheus"
PROM_DATA_DIR="$PROJECT_DIR/tools/prometheus-data"
PROM_CONFIG="$PROJECT_DIR/tools/prometheus_bgai.yml"

if [[ ! -x "$PROM_BIN" ]]; then
    echo "       WARNING: Prometheus not installed. Run ./scripts/prometheus_setup.sh first"
    PROMETHEUS_PID=""
else
    pkill -f "$PROM_BIN" 2>/dev/null || true
    mkdir -p "$PROM_DATA_DIR"
    "$PROM_BIN" \
        --config.file="$PROM_CONFIG" \
        --storage.tsdb.path="$PROM_DATA_DIR" \
        --web.listen-address="0.0.0.0:9090" \
        --web.enable-lifecycle \
        > "$LOG_DIR/prometheus_$TIMESTAMP.log" 2>&1 &
    PROMETHEUS_PID=$!
    echo "       Prometheus PID: $PROMETHEUS_PID"
    sleep 2
fi

# =============================================================================
# Start Grafana
# =============================================================================
echo "       Starting Grafana..."
GRAFANA_DIR="$PROJECT_DIR/tools/grafana-11.3.0"
if [[ -d "$GRAFANA_DIR" ]]; then
    pkill -f "grafana-server" 2>/dev/null || true
    "$GRAFANA_DIR/bin/grafana-server" \
        --homepath="$GRAFANA_DIR" \
        --config="$GRAFANA_DIR/conf/custom.ini" \
        > "$LOG_DIR/grafana_$TIMESTAMP.log" 2>&1 &
    GRAFANA_PID=$!
    echo "       Grafana PID: $GRAFANA_PID"
else
    echo "       WARNING: Grafana not installed. Run ./scripts/grafana_setup.sh first"
    GRAFANA_PID=""
fi
sleep 2

# =============================================================================
# Start discovery updater
# =============================================================================
echo "       Starting Prometheus discovery updater..."
DISCOVERY_PATH="$PROJECT_DIR/tools/bgai_prometheus_targets.json"
CUDA_VISIBLE_DEVICES="" python -c "
from distributed.metrics import start_discovery_updater
import time
start_discovery_updater(
    redis_host='$REDIS_HOST',
    redis_port=$REDIS_PORT,
    redis_password='$REDIS_PASSWORD',
    update_interval=15,
    output_path='$DISCOVERY_PATH',
)
print(f'Discovery updater started, writing to $DISCOVERY_PATH')
while True:
    time.sleep(3600)
" > "$LOG_DIR/discovery_updater_$TIMESTAMP.log" 2>&1 &
DISCOVERY_PID=$!
sleep 2

# =============================================================================
# Coordinator
# =============================================================================
echo "[5/7] Starting coordinator..."
CUDA_VISIBLE_DEVICES="" python -m distributed.cli.main coordinator \
    --config-file "$CONFIG_FILE" \
    --dashboard \
    > "$LOG_DIR/coordinator_$TIMESTAMP.log" 2>&1 &
COORD_PID=$!
sleep 5

if ! kill -0 $COORD_PID 2>/dev/null; then
    echo "ERROR: Coordinator failed to start. Check log:"
    tail -20 "$LOG_DIR/coordinator_$TIMESTAMP.log"
    exit 1
fi

# =============================================================================
# Workers on head
# =============================================================================
echo "[6/7] Starting training worker..."
# Use fixed GPU memory caps (GB) so allocation is stable across GPUs.
python -m distributed.cli.main training-worker \
    --config-file "$CONFIG_FILE" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --gpu-mem-gb "$TRAIN_MEM_GB" \
    ${HEAD_TRAIN_BATCH_SIZE:+--batch-size $HEAD_TRAIN_BATCH_SIZE} \
    > "$LOG_DIR/training_$TIMESTAMP.log" 2>&1 &
TRAIN_PID=$!

echo "       Starting game worker..."
python -m distributed.cli.main game-worker \
    --config-file "$CONFIG_FILE" \
    --gpu-mem-gb "$GAME_MEM_GB" \
    ${HEAD_GAME_BATCH_SIZE:+--batch-size $HEAD_GAME_BATCH_SIZE} \
    > "$LOG_DIR/game_gpu_$TIMESTAMP.log" 2>&1 &
GAME_PID=$!

echo "[7/7] Starting eval worker (CPU)..."
JAX_PLATFORMS="$EVAL_JAX_PLATFORMS" python -m distributed.cli.main eval-worker \
    --config-file "$CONFIG_FILE" \
    --eval-games 50 \
    --eval-types "random,self_play" \
    --num-gpus 0 \
    > "$LOG_DIR/eval_$TIMESTAMP.log" 2>&1 &
EVAL_PID=$!

echo ""
echo "=============================================="
echo "  All services started!"
echo "=============================================="
echo "PIDs: MLflow=$MLFLOW_PID Prometheus=$PROMETHEUS_PID Grafana=$GRAFANA_PID Discovery=$DISCOVERY_PID Coord=$COORD_PID Train=$TRAIN_PID Game=$GAME_PID Eval=$EVAL_PID"
echo "Grafana:    http://$HEAD_IP:3000"
echo "Prometheus: http://$HEAD_IP:9090"
echo "MLflow:     http://$HEAD_IP:5000"
echo ""
echo "Check status: ./scripts/status.sh"
echo "Stop all:     ./scripts/stop_all.sh"
