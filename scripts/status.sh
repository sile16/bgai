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

# Try Tailscale IP first, fallback to local IP
TAILSCALE_IP="100.105.50.111"
LOCAL_IP="192.168.20.40"

if ping -c 1 -W 1 "$TAILSCALE_IP" &>/dev/null; then
    HEAD_IP="$TAILSCALE_IP"
else
    HEAD_IP="$LOCAL_IP"
fi
RAY_CLIENT_PORT="10001"

python -m distributed.cli.main status \
    --coordinator-address "ray://$HEAD_IP:$RAY_CLIENT_PORT"
