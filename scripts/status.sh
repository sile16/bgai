#!/bin/bash
# Check cluster status

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate virtual environment
if [[ -d "$PROJECT_DIR/venv-bgai" ]]; then
    source "$PROJECT_DIR/venv-bgai/bin/activate"
elif [[ -d "$PROJECT_DIR/venv" ]]; then
    source "$PROJECT_DIR/venv/bin/activate"
fi

# Load config
CONFIG_FILE="$PROJECT_DIR/configs/distributed.yaml"

echo "=== BGAI Cluster Status ==="
python -m distributed.cli.main status \
    --config-file "$CONFIG_FILE" 2>/dev/null || echo "No workers registered (start with ./scripts/start_all_head.sh)"
