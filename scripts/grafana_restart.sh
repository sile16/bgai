#!/bin/bash
# Restart Grafana for BGAI metrics

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GRAFANA_VERSION="11.3.0"
GRAFANA_DIR="$PROJECT_DIR/tools/grafana-$GRAFANA_VERSION"

# Check if Grafana is installed
if [[ ! -d "$GRAFANA_DIR" ]]; then
    echo "Grafana not found at $GRAFANA_DIR"
    echo "Run scripts/grafana_setup.sh first to install Grafana"
    exit 1
fi

# Stop existing Grafana
echo "Stopping existing Grafana..."
pkill -f -- "--homepath=$GRAFANA_DIR" 2>/dev/null || true
pkill -f -- "--config=$GRAFANA_DIR/conf/custom.ini" 2>/dev/null || true
pkill -f "grafana-server" 2>/dev/null || true
sleep 1

# Force kill if still running on port 3000
if ss -tlnp 2>/dev/null | grep -q ":3000"; then
    echo "Force killing process on port 3000..."
    fuser -k 3000/tcp 2>/dev/null || true
    sleep 1
fi

# Start Grafana
echo "Starting Grafana..."
nohup "$GRAFANA_DIR/bin/grafana-server" \
    --homepath="$GRAFANA_DIR" \
    --config="$GRAFANA_DIR/conf/custom.ini" \
    > "$PROJECT_DIR/logs/grafana.log" 2>&1 &

GRAFANA_PID=$!
echo "Grafana started with PID $GRAFANA_PID"

# Wait for startup
sleep 2

# Check if running
if kill -0 $GRAFANA_PID 2>/dev/null; then
    echo "Grafana is running at http://localhost:3000"
else
    echo "Grafana failed to start. Check logs/grafana.log"
    exit 1
fi
