#!/bin/bash
# Stop all Ray processes

echo "Stopping Ray cluster..."
ray stop --force 2>/dev/null || true

echo "Killing any remaining Python processes from distributed training..."
pkill -f "distributed.cli.main" 2>/dev/null || true

echo "Done."
