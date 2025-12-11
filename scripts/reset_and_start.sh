#!/bin/bash
# Reset and start BGAI distributed training with explicit run mode selection
#
# Usage: ./scripts/reset_and_start.sh [--new|--continue]
#   --new:      Start a completely fresh run (clears all data)
#   --continue: Continue existing run (keeps data, just restarts services)
#   No args:    Interactive prompt to choose

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# =============================================================================
# Colors for output
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Load configuration
# =============================================================================
CONFIG_FILE="$PROJECT_DIR/configs/distributed.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}ERROR: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

REDIS_HOST=$(grep -A5 "^redis:" "$CONFIG_FILE" | grep "host:" | head -1 | sed 's/.*: *"\([^"]*\)".*/\1/')
REDIS_PORT=$(grep -A5 "^redis:" "$CONFIG_FILE" | grep "port:" | sed 's/.*: *\([0-9]*\).*/\1/')
REDIS_PASSWORD=$(grep -A5 "^redis:" "$CONFIG_FILE" | grep "password:" | sed 's/.*: *"\([^"]*\)".*/\1/')

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-bgai-password}"

# =============================================================================
# Parse arguments
# =============================================================================
RUN_MODE=""
if [[ "$1" == "--new" ]]; then
    RUN_MODE="new"
elif [[ "$1" == "--continue" ]]; then
    RUN_MODE="continue"
fi

# =============================================================================
# Interactive prompt if no mode specified
# =============================================================================
if [[ -z "$RUN_MODE" ]]; then
    echo ""
    echo -e "${BLUE}=============================================="
    echo "  BGAI Training Run Mode Selection"
    echo "==============================================${NC}"
    echo ""

    # Check current state from Redis
    CURRENT_RUN_ID=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning GET bgai:run:id 2>/dev/null || echo "")
    CURRENT_VERSION=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning GET bgai:model:version 2>/dev/null || echo "0")

    if [[ -n "$CURRENT_RUN_ID" && "$CURRENT_RUN_ID" != "(nil)" ]]; then
        echo -e "  Current run ID: ${GREEN}$CURRENT_RUN_ID${NC}"
        echo -e "  Model version:  ${GREEN}$CURRENT_VERSION${NC}"
        echo ""
        echo -e "${YELLOW}Choose an option:${NC}"
        echo "  1) ${GREEN}CONTINUE${NC} - Resume existing run (keep all data)"
        echo "  2) ${RED}NEW${NC}      - Start fresh run (clear all data)"
        echo ""
        read -p "Enter choice [1/2]: " choice
        case $choice in
            1) RUN_MODE="continue" ;;
            2) RUN_MODE="new" ;;
            *) echo "Invalid choice. Exiting."; exit 1 ;;
        esac
    else
        echo -e "  ${YELLOW}No existing run found.${NC}"
        echo ""
        RUN_MODE="new"
        echo -e "  Starting ${GREEN}NEW${NC} run..."
    fi
    echo ""
fi

# =============================================================================
# Stop all existing services
# =============================================================================
echo -e "${BLUE}[1/5] Stopping all existing services...${NC}"
"$SCRIPT_DIR/stop_all.sh"
sleep 2

# =============================================================================
# Handle NEW run - clear all data
# =============================================================================
if [[ "$RUN_MODE" == "new" ]]; then
    echo -e "${BLUE}[2/5] Clearing all previous data for NEW run...${NC}"

    # Clear Redis data
    echo "  - Clearing Redis (all bgai:* keys)..."
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning KEYS "bgai:*" | \
        xargs -r redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning DEL 2>/dev/null || true

    # Clear replay buffer
    echo "  - Clearing replay buffer..."
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning KEYS "buffer:*" | \
        xargs -r redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning DEL 2>/dev/null || true

    # Clear checkpoints
    echo "  - Clearing checkpoints..."
    rm -rf "$PROJECT_DIR/checkpoints/"* 2>/dev/null || true

    # Clear old logs
    echo "  - Clearing old logs..."
    rm -f "$PROJECT_DIR/logs/"*.log 2>/dev/null || true

    # Generate new run ID with timestamp
    RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
    echo ""
    echo -e "  ${GREEN}New run ID: $RUN_ID${NC}"

    # Set the run ID in Redis so workers pick it up
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning SET bgai:run:id "$RUN_ID" >/dev/null
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning SET bgai:run:status "running" >/dev/null
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning SET bgai:model:version "0" >/dev/null

else
    echo -e "${BLUE}[2/5] CONTINUE mode - keeping existing data...${NC}"
    CURRENT_RUN_ID=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning GET bgai:run:id 2>/dev/null)
    CURRENT_VERSION=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning GET bgai:model:version 2>/dev/null)
    echo -e "  Run ID: ${GREEN}$CURRENT_RUN_ID${NC}"
    echo -e "  Model version: ${GREEN}$CURRENT_VERSION${NC}"

    # Ensure run status is set to running
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning SET bgai:run:status "running" >/dev/null
fi

# =============================================================================
# Clear stale worker registrations (for both modes)
# =============================================================================
echo -e "${BLUE}[3/5] Clearing stale worker registrations...${NC}"
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning KEYS "bgai:workers:*" | \
    xargs -r redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning DEL 2>/dev/null || true

# =============================================================================
# Start all services
# =============================================================================
echo -e "${BLUE}[4/5] Starting all services...${NC}"
echo ""
"$SCRIPT_DIR/start_all_head.sh"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}=============================================="
echo "  BGAI Training Started Successfully!"
echo "==============================================${NC}"
echo ""
if [[ "$RUN_MODE" == "new" ]]; then
    echo -e "  Mode: ${GREEN}NEW RUN${NC}"
    echo -e "  Run ID: ${GREEN}$RUN_ID${NC}"
    echo "  All previous data cleared."
else
    echo -e "  Mode: ${YELLOW}CONTINUE${NC}"
    echo -e "  Run ID: ${GREEN}$CURRENT_RUN_ID${NC}"
    echo "  Existing data preserved."
fi
echo ""
echo -e "${BLUE}[5/5] To monitor logs, run:${NC}"
echo "  tail -f logs/*.log"
echo ""
