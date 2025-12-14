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

echo "Stopping Pushgateway..."
pkill -f "pushgateway" 2>/dev/null || true

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
    for worker_type in game eval training; do
        for pid_file in "$LOG_DIR"/${worker_type}_*.pid; do
            if [[ -f "$pid_file" ]]; then
                # Derive stop file from pid file name
                stop_file="${pid_file%.pid}.stop"
                touch "$stop_file"
                echo "  Created stop file: $stop_file"

                # Also kill the wrapper bash loop by PID
                wrapper_pid=$(cat "$pid_file" 2>/dev/null)
                if [[ -n "$wrapper_pid" ]] && kill -0 "$wrapper_pid" 2>/dev/null; then
                    echo "  Killing wrapper process: $wrapper_pid"
                    # Kill the entire process group to get the bash loop and its children
                    kill -- -"$wrapper_pid" 2>/dev/null || kill "$wrapper_pid" 2>/dev/null || true
                fi
            fi
        done
    done
fi

# Kill the Python worker processes directly (in case wrapper kill missed them)
echo "Killing Python worker processes..."
pkill -f "distributed.cli.main game-worker" 2>/dev/null || true
pkill -f "distributed.cli.main eval-worker" 2>/dev/null || true
pkill -f "distributed.cli.main training-worker" 2>/dev/null || true

# Wait for graceful shutdown
sleep 2

# Force kill any remaining worker processes
if pgrep -f "distributed.cli.main" >/dev/null 2>&1; then
    echo "Force killing remaining workers..."
    pkill -9 -f "distributed.cli.main" 2>/dev/null || true
fi

# Clean up pid files (but keep stop files a bit longer to prevent restart races)
if [[ -d "$LOG_DIR" ]]; then
    rm -f "$LOG_DIR"/*.pid 2>/dev/null || true
fi

# Wait a moment then clean up stop files
sleep 1
if [[ -d "$LOG_DIR" ]]; then
    rm -f "$LOG_DIR"/*.stop 2>/dev/null || true
fi

echo "Stopping discovery updater..."
pkill -f "start_discovery_updater" 2>/dev/null || true

echo "Done."
