#!/bin/bash
# Download and setup Prometheus Pushgateway for BGAI metrics pushing (e.g. Colab).
# This only needs to be run once to install Pushgateway locally under tools/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PUSHGATEWAY_VERSION="${PUSHGATEWAY_VERSION:-1.9.0}"
PUSHGATEWAY_DIR="$PROJECT_DIR/tools/pushgateway-$PUSHGATEWAY_VERSION"
PUSHGATEWAY_TARBALL="pushgateway-${PUSHGATEWAY_VERSION}.linux-amd64.tar.gz"

echo "Setting up Pushgateway $PUSHGATEWAY_VERSION..."

if [[ ! -d "$PUSHGATEWAY_DIR" ]]; then
    echo "Downloading Pushgateway..."
    cd "$PROJECT_DIR/tools"
    rm -f pushgateway.tar.gz
    wget -q "https://github.com/prometheus/pushgateway/releases/download/v${PUSHGATEWAY_VERSION}/${PUSHGATEWAY_TARBALL}" -O pushgateway.tar.gz
    tar -xzf pushgateway.tar.gz
    mv "pushgateway-${PUSHGATEWAY_VERSION}.linux-amd64" "pushgateway-${PUSHGATEWAY_VERSION}"
    rm pushgateway.tar.gz
    echo "Pushgateway downloaded to $PUSHGATEWAY_DIR"
else
    echo "Pushgateway already installed at $PUSHGATEWAY_DIR"
fi

# Create/Update symlink
ln -sfn "$PUSHGATEWAY_DIR" "$PROJECT_DIR/tools/pushgateway"

echo "Pushgateway setup complete!"
echo "  Installation: $PUSHGATEWAY_DIR"
echo ""
echo "To start Pushgateway manually:"
echo "  $PROJECT_DIR/tools/pushgateway/pushgateway --web.listen-address=0.0.0.0:9091"

