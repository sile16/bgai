#!/bin/bash
# Download and setup Grafana for Ray metrics
# This only needs to be run once to install Grafana

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GRAFANA_VERSION="11.3.0"
GRAFANA_DIR="$PROJECT_DIR/tools/grafana-$GRAFANA_VERSION"

echo "Setting up Grafana $GRAFANA_VERSION..."

# Download if not present
if [[ ! -d "$GRAFANA_DIR" ]]; then
    echo "Downloading Grafana..."
    mkdir -p "$PROJECT_DIR/tools"
    cd "$PROJECT_DIR/tools"
    wget -q "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz" -O grafana.tar.gz
    tar -xzf grafana.tar.gz
    mv "grafana-v${GRAFANA_VERSION}" "grafana-${GRAFANA_VERSION}"
    rm grafana.tar.gz
    echo "Grafana downloaded to $GRAFANA_DIR"
fi

# Create required directories
mkdir -p "$GRAFANA_DIR/data"
mkdir -p "$GRAFANA_DIR/conf/provisioning/datasources"
mkdir -p "$GRAFANA_DIR/conf/provisioning/dashboards"

# Ray generates dashboards at runtime in the session directory
# These are processed templates with proper PromQL queries
RAY_DASHBOARDS_DIR="/tmp/ray/session_latest/metrics/grafana/dashboards"

echo "Using Ray generated dashboards from: $RAY_DASHBOARDS_DIR"
echo "Note: Ray must be running for dashboards to be available"

# Create Prometheus datasource config
cat > "$GRAFANA_DIR/conf/provisioning/datasources/prometheus.yaml" << EOF
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
    editable: true
EOF

# Create dashboard provisioning config
# Points to Ray's runtime-generated dashboards (processed templates)
cat > "$GRAFANA_DIR/conf/provisioning/dashboards/ray.yaml" << EOF
apiVersion: 1
providers:
  - name: 'Ray'
    orgId: 1
    folder: 'Ray'
    type: file
    disableDeletion: false
    editable: true
    updateIntervalSeconds: 30
    options:
      path: $RAY_DASHBOARDS_DIR
EOF

# Create custom ini file for anonymous access
cat > "$GRAFANA_DIR/conf/custom.ini" << EOF
[server]
http_port = 3000
root_url = http://localhost:3000

[auth.anonymous]
enabled = true
org_name = Main Org.
org_role = Viewer

[security]
allow_embedding = true

[paths]
data = $GRAFANA_DIR/data
plugins = $GRAFANA_DIR/data/plugins
provisioning = $GRAFANA_DIR/conf/provisioning
EOF

echo ""
echo "Grafana setup complete!"
echo "  Installation: $GRAFANA_DIR"
echo ""
echo "To start Grafana manually:"
echo "  $GRAFANA_DIR/bin/grafana-server --homepath=$GRAFANA_DIR --config=$GRAFANA_DIR/conf/custom.ini"
echo ""
echo "Or use the start_all_head.sh script which includes Grafana startup."
