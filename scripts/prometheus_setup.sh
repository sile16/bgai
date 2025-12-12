#!/bin/bash
# Download and setup Prometheus for BGAI metrics
# This only needs to be run once to install Prometheus locally under tools/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROMETHEUS_VERSION="3.8.0"
PROMETHEUS_DIR="$PROJECT_DIR/tools/prometheus-$PROMETHEUS_VERSION"
PROMETHEUS_TARBALL="prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz"

echo "Setting up Prometheus $PROMETHEUS_VERSION..."

if [[ ! -d "$PROMETHEUS_DIR" ]]; then
    echo "Downloading Prometheus..."
    mkdir -p "$PROJECT_DIR/tools"
    cd "$PROJECT_DIR/tools"
    wget -q "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/${PROMETHEUS_TARBALL}" -O prometheus.tar.gz
    tar -xzf prometheus.tar.gz
    mv "prometheus-${PROMETHEUS_VERSION}.linux-amd64" "prometheus-${PROMETHEUS_VERSION}"
    rm prometheus.tar.gz
    echo "Prometheus downloaded to $PROMETHEUS_DIR"
fi

# Create/refresh stable symlink
ln -sfn "$PROMETHEUS_DIR" "$PROJECT_DIR/tools/prometheus"

# Create data dir for local TSDB
mkdir -p "$PROJECT_DIR/tools/prometheus-data"

echo ""
echo "Prometheus setup complete!"
echo "  Installation: $PROMETHEUS_DIR"
echo ""
echo "To start Prometheus manually:"
echo "  $PROJECT_DIR/tools/prometheus/prometheus --config.file=$PROJECT_DIR/tools/prometheus_bgai.yml --storage.tsdb.path=$PROJECT_DIR/tools/prometheus-data"
echo ""
echo "Or use the head_start.sh script which includes Prometheus startup."
