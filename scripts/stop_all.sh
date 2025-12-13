#!/bin/bash
# Stop all BGAI distributed training processes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

echo "Stopping Grafana..."
# Only kill Grafana instances started from this repo (match args, not argv0).
pkill -f -- "--homepath=$PROJECT_DIR/tools/grafana-" 2>/dev/null || true
pkill -f -- "--config=$PROJECT_DIR/tools/grafana-" 2>/dev/null || true

echo "Stopping Prometheus..."
pkill -f "prometheus" 2>/dev/null || true

echo "Stopping MLFlow..."
pkill -f "mlflow server" 2>/dev/null || true
pkill -f "uvicorn.*mlflow" 2>/dev/null || true

echo "Stopping distributed training workers..."

# Create stop files for worker_start.sh restart loops
# This signals the wrapper scripts to exit gracefully
if [[ -d "$LOG_DIR" ]]; then
    for stop_file in "$LOG_DIR"/*.stop; do
        # Touch any existing stop files (from previous runs)
        [[ -e "$stop_file" ]] && touch "$stop_file"
    done

    # Create stop files for all possible worker types and devices
    for worker_type in game eval; do
        for pid_file in "$LOG_DIR"/${worker_type}_*.pid; do
            if [[ -f "$pid_file" ]]; then
                # Derive stop file from pid file name
                stop_file="${pid_file%.pid}.stop"
                touch "$stop_file"
                echo "  Created stop file: $stop_file"
            fi
        done
    done
fi

# Kill the Python worker processes
pkill -f "distributed.cli.main" 2>/dev/null || true

# Wait briefly for graceful shutdown
sleep 1

# Kill any remaining wrapper bash loops (from worker_start.sh)
# These are bash processes that contain the restart loop
pkill -f "Starting game worker" 2>/dev/null || true
pkill -f "Starting eval worker" 2>/dev/null || true

# Clean up pid files
if [[ -d "$LOG_DIR" ]]; then
    rm -f "$LOG_DIR"/*.pid 2>/dev/null || true
    rm -f "$LOG_DIR"/*.stop 2>/dev/null || true
fi

echo "Stopping discovery updater..."
pkill -f "start_discovery_updater" 2>/dev/null || true

echo "Done."
