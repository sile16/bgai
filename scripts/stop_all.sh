#!/bin/bash
# Stop all Ray processes and related services

echo "Stopping Grafana..."
pkill -f "grafana-server" 2>/dev/null || true

echo "Stopping Prometheus metrics..."
ray metrics shutdown-prometheus 2>/dev/null || true

echo "Stopping Ray cluster..."
ray stop --force 2>/dev/null || true

echo "Killing any remaining Python processes from distributed training..."
pkill -f "distributed.cli.main" 2>/dev/null || true

echo "Done."
