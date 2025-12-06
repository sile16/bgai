#!/bin/bash
# Stop all BGAI distributed training processes

echo "Stopping Grafana..."
pkill -f "grafana-server" 2>/dev/null || true

echo "Stopping Prometheus..."
pkill -f "prometheus" 2>/dev/null || true

echo "Stopping MLFlow..."
pkill -f "mlflow server" 2>/dev/null || true
pkill -f "uvicorn.*mlflow" 2>/dev/null || true

echo "Stopping distributed training workers..."
pkill -f "distributed.cli.main" 2>/dev/null || true

echo "Stopping discovery updater..."
pkill -f "start_discovery_updater" 2>/dev/null || true

echo "Done."
