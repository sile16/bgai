#!/bin/bash
# Stop all BGAI distributed training processes

echo "Stopping Grafana..."
pkill -f "grafana-server" 2>/dev/null || true

echo "Stopping Prometheus..."
pkill -f "prometheus" 2>/dev/null || true

echo "Stopping distributed training workers..."
pkill -f "distributed.cli.main" 2>/dev/null || true

echo "Done."
