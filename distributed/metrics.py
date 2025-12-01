"""Prometheus metrics for distributed training.

This module provides Prometheus metrics that can be scraped by the existing
Prometheus instance started by Ray. Metrics are exposed on a separate HTTP
port from each worker.

Workers register their metrics endpoints in Redis for dynamic discovery.
A background thread updates the Prometheus service discovery file.

Usage:
    from distributed.metrics import get_metrics, start_metrics_server

    # Start metrics server on worker initialization
    start_metrics_server(port=9100, worker_id='gpu-0', worker_type='game',
                         redis_host='localhost', redis_port=6379)

    # Get metrics instance and record values
    metrics = get_metrics()
    metrics.games_total.labels(worker_id='gpu-0').inc()
    metrics.training_loss.set(0.5)
"""

import json
import os
import socket
import threading
import time
from typing import Optional, Dict, Any
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    CollectorRegistry,
    start_http_server,
    REGISTRY,
)


class BGAIMetrics:
    """Container for all BGAI training metrics."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics.

        Args:
            registry: Optional custom registry. Uses default if None.
        """
        self.registry = registry or REGISTRY

        # =================================================================
        # Game Collection Metrics
        # =================================================================
        self.games_total = Counter(
            'bgai_games_total',
            'Total games completed',
            ['worker_id', 'worker_type'],
            registry=self.registry,
        )

        self.experiences_total = Counter(
            'bgai_experiences_total',
            'Total experiences collected',
            ['worker_id'],
            registry=self.registry,
        )

        self.steps_total = Counter(
            'bgai_collection_steps_total',
            'Total collection steps (environment steps)',
            ['worker_id'],
            registry=self.registry,
        )

        self.collection_steps_per_second = Gauge(
            'bgai_collection_steps_per_second',
            'Current collection rate (steps/second)',
            ['worker_id'],
            registry=self.registry,
        )

        self.games_per_minute = Gauge(
            'bgai_games_per_minute',
            'Current game completion rate',
            ['worker_id'],
            registry=self.registry,
        )

        self.episode_length = Histogram(
            'bgai_episode_length',
            'Distribution of episode lengths (steps)',
            ['worker_id'],
            buckets=[50, 100, 150, 200, 250, 300, 400, 500, 750, 1000],
            registry=self.registry,
        )

        # =================================================================
        # Training Metrics
        # =================================================================
        self.training_steps_total = Counter(
            'bgai_training_steps_total',
            'Total training gradient steps',
            ['worker_id'],
            registry=self.registry,
        )

        self.training_batches_total = Counter(
            'bgai_training_batches_total',
            'Total training batches (collection-gated)',
            ['worker_id'],
            registry=self.registry,
        )

        self.training_loss = Gauge(
            'bgai_training_loss',
            'Current training loss',
            ['worker_id', 'loss_type'],
            registry=self.registry,
        )

        self.training_steps_per_second = Gauge(
            'bgai_training_steps_per_second',
            'Current training rate (steps/second)',
            ['worker_id'],
            registry=self.registry,
        )

        self.training_batch_duration = Histogram(
            'bgai_training_batch_duration_seconds',
            'Duration of training batches',
            ['worker_id'],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
            registry=self.registry,
        )

        self.training_batch_steps = Histogram(
            'bgai_training_batch_steps',
            'Number of steps per training batch',
            ['worker_id'],
            buckets=[10, 50, 100, 200, 500, 1000, 2000],
            registry=self.registry,
        )

        # =================================================================
        # Buffer Metrics
        # =================================================================
        self.buffer_size = Gauge(
            'bgai_buffer_size',
            'Current replay buffer size (experiences)',
            registry=self.registry,
        )

        self.buffer_games = Gauge(
            'bgai_buffer_games',
            'Current number of games in buffer',
            registry=self.registry,
        )

        self.games_since_last_train = Gauge(
            'bgai_games_since_last_train',
            'Games collected since last training batch',
            ['worker_id'],
            registry=self.registry,
        )

        self.buffer_episodes = Gauge(
            'bgai_buffer_episodes',
            'Current number of episodes in buffer',
            registry=self.registry,
        )

        # =================================================================
        # Surprise Sampling Metrics
        # =================================================================
        self.surprise_score = Histogram(
            'bgai_surprise_score',
            'Distribution of episode surprise scores',
            ['worker_id'],
            buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0],
            registry=self.registry,
        )

        self.surprise_score_max = Gauge(
            'bgai_surprise_score_max',
            'Maximum surprise score in buffer',
            registry=self.registry,
        )

        self.surprise_score_mean = Gauge(
            'bgai_surprise_score_mean',
            'Mean surprise score in buffer',
            registry=self.registry,
        )

        self.episodes_with_surprise = Gauge(
            'bgai_episodes_with_surprise',
            'Number of episodes with surprise scores',
            registry=self.registry,
        )

        # =================================================================
        # Model Metrics
        # =================================================================
        self.model_version = Gauge(
            'bgai_model_version',
            'Current model version',
            ['worker_id', 'worker_type'],
            registry=self.registry,
        )

        self.weight_updates_total = Counter(
            'bgai_weight_updates_total',
            'Total weight updates pushed',
            ['worker_id'],
            registry=self.registry,
        )

        # =================================================================
        # Evaluation Metrics
        # =================================================================
        self.eval_games_total = Counter(
            'bgai_eval_games_total',
            'Total evaluation games played',
            ['worker_id', 'eval_type'],
            registry=self.registry,
        )

        self.eval_win_rate = Gauge(
            'bgai_eval_win_rate',
            'Win rate from most recent evaluation',
            ['eval_type', 'model_version'],
            registry=self.registry,
        )

        self.eval_wins = Counter(
            'bgai_eval_wins_total',
            'Total evaluation wins',
            ['worker_id', 'eval_type'],
            registry=self.registry,
        )

        self.eval_losses = Counter(
            'bgai_eval_losses_total',
            'Total evaluation losses',
            ['worker_id', 'eval_type'],
            registry=self.registry,
        )

        self.eval_duration = Histogram(
            'bgai_eval_duration_seconds',
            'Duration of evaluation runs',
            ['worker_id', 'eval_type'],
            buckets=[10, 30, 60, 120, 300, 600, 1200, 1800],
            registry=self.registry,
        )

        self.eval_avg_game_length = Gauge(
            'bgai_eval_avg_game_length',
            'Average game length in most recent evaluation',
            ['eval_type', 'model_version'],
            registry=self.registry,
        )

        self.eval_runs_total = Counter(
            'bgai_eval_runs_total',
            'Total evaluation runs completed',
            ['worker_id', 'eval_type'],
            registry=self.registry,
        )

        # =================================================================
        # Worker Info
        # =================================================================
        self.worker_info = Info(
            'bgai_worker',
            'Worker information',
            ['worker_id'],
            registry=self.registry,
        )

        self.worker_status = Gauge(
            'bgai_worker_status',
            'Worker status (1=running, 0=stopped)',
            ['worker_id', 'worker_type'],
            registry=self.registry,
        )


# Global metrics instance
_metrics: Optional[BGAIMetrics] = None
_metrics_lock = threading.Lock()
_server_started = False


def get_metrics() -> BGAIMetrics:
    """Get the global metrics instance.

    Returns:
        BGAIMetrics instance (creates one if needed).
    """
    global _metrics
    with _metrics_lock:
        if _metrics is None:
            _metrics = BGAIMetrics()
        return _metrics


def start_metrics_server(port: int = 9100) -> bool:
    """Start the Prometheus metrics HTTP server.

    Args:
        port: Port to serve metrics on.

    Returns:
        True if server started, False if already running.
    """
    global _server_started
    with _metrics_lock:
        if _server_started:
            return False

        try:
            start_http_server(port)
            _server_started = True
            print(f"Prometheus metrics server started on port {port}")
            return True
        except Exception as e:
            print(f"Failed to start metrics server on port {port}: {e}")
            # Try alternative ports
            for alt_port in range(port + 1, port + 10):
                try:
                    start_http_server(alt_port)
                    _server_started = True
                    print(f"Prometheus metrics server started on port {alt_port}")
                    return True
                except Exception:
                    continue
            return False


def reset_metrics():
    """Reset the global metrics instance (for testing)."""
    global _metrics, _server_started
    with _metrics_lock:
        _metrics = None
        _server_started = False


# =============================================================================
# Worker Registration for Dynamic Discovery
# =============================================================================

WORKER_REGISTRY_KEY = "bgai:metrics:workers"
DISCOVERY_FILE_PATH = "/tmp/bgai_prometheus_targets.json"


def _get_worker_ip() -> str:
    """Get the IP address of this worker."""
    try:
        # Try to get the IP that would be used to connect to an external host
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def register_metrics_endpoint(
    redis_client,
    worker_id: str,
    worker_type: str,
    port: int,
    ttl_seconds: int = 60,
) -> None:
    """Register this worker's metrics endpoint in Redis.

    Args:
        redis_client: Redis client instance.
        worker_id: Unique worker identifier.
        worker_type: Type of worker ('game' or 'training').
        port: Port the metrics server is listening on.
        ttl_seconds: Time-to-live for the registration (for cleanup).
    """
    ip = _get_worker_ip()
    endpoint = f"{ip}:{port}"

    worker_info = {
        "worker_id": worker_id,
        "worker_type": worker_type,
        "endpoint": endpoint,
        "registered_at": time.time(),
    }

    # Store in Redis hash with TTL-based cleanup
    redis_client.hset(WORKER_REGISTRY_KEY, worker_id, json.dumps(worker_info))

    # Also set an expiring key for heartbeat tracking
    redis_client.setex(f"bgai:metrics:heartbeat:{worker_id}", ttl_seconds, endpoint)


def unregister_metrics_endpoint(redis_client, worker_id: str) -> None:
    """Unregister this worker's metrics endpoint from Redis.

    Args:
        redis_client: Redis client instance.
        worker_id: Unique worker identifier.
    """
    redis_client.hdel(WORKER_REGISTRY_KEY, worker_id)
    redis_client.delete(f"bgai:metrics:heartbeat:{worker_id}")


def get_registered_workers(redis_client) -> Dict[str, Any]:
    """Get all registered worker metrics endpoints.

    Args:
        redis_client: Redis client instance.

    Returns:
        Dict mapping worker_id to worker info.
    """
    workers = {}
    all_workers = redis_client.hgetall(WORKER_REGISTRY_KEY)

    for worker_id_bytes, info_bytes in all_workers.items():
        worker_id = worker_id_bytes.decode() if isinstance(worker_id_bytes, bytes) else worker_id_bytes
        info_str = info_bytes.decode() if isinstance(info_bytes, bytes) else info_bytes

        try:
            info = json.loads(info_str)

            # Check if heartbeat is still alive
            heartbeat_key = f"bgai:metrics:heartbeat:{worker_id}"
            if redis_client.exists(heartbeat_key):
                workers[worker_id] = info
            else:
                # Cleanup stale registration
                redis_client.hdel(WORKER_REGISTRY_KEY, worker_id)
        except json.JSONDecodeError:
            continue

    return workers


def update_prometheus_discovery_file(
    redis_client,
    output_path: str = DISCOVERY_FILE_PATH,
) -> int:
    """Update the Prometheus file-based service discovery file.

    Args:
        redis_client: Redis client instance.
        output_path: Path to write the discovery JSON file.

    Returns:
        Number of workers in the discovery file.
    """
    workers = get_registered_workers(redis_client)

    # Group by worker type
    game_targets = []
    training_targets = []

    for worker_id, info in workers.items():
        endpoint = info.get("endpoint", "")
        worker_type = info.get("worker_type", "unknown")

        if worker_type == "game":
            game_targets.append(endpoint)
        elif worker_type == "training":
            training_targets.append(endpoint)

    # Build Prometheus file_sd_configs format
    discovery = []

    if game_targets:
        discovery.append({
            "labels": {"job": "bgai_game_workers", "worker_type": "game"},
            "targets": game_targets,
        })

    if training_targets:
        discovery.append({
            "labels": {"job": "bgai_training_workers", "worker_type": "training"},
            "targets": training_targets,
        })

    # Write atomically
    temp_path = output_path + ".tmp"
    with open(temp_path, "w") as f:
        json.dump(discovery, f, indent=2)
    os.replace(temp_path, output_path)

    return len(workers)


class MetricsDiscoveryUpdater(threading.Thread):
    """Background thread that periodically updates the Prometheus discovery file."""

    def __init__(
        self,
        redis_host: str,
        redis_port: int,
        redis_password: Optional[str] = None,
        update_interval: int = 15,
        output_path: str = DISCOVERY_FILE_PATH,
    ):
        super().__init__(daemon=True)
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.update_interval = update_interval
        self.output_path = output_path
        self._stop_event = threading.Event()
        self._redis = None

    def _get_redis(self):
        if self._redis is None:
            import redis
            self._redis = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
            )
        return self._redis

    def run(self):
        while not self._stop_event.is_set():
            try:
                redis_client = self._get_redis()
                num_workers = update_prometheus_discovery_file(
                    redis_client, self.output_path
                )
                # print(f"Updated Prometheus discovery: {num_workers} workers")
            except Exception as e:
                print(f"Error updating Prometheus discovery: {e}")

            self._stop_event.wait(self.update_interval)

    def stop(self):
        self._stop_event.set()


# Global discovery updater instance
_discovery_updater: Optional[MetricsDiscoveryUpdater] = None


def start_discovery_updater(
    redis_host: str,
    redis_port: int,
    redis_password: Optional[str] = None,
    update_interval: int = 15,
) -> MetricsDiscoveryUpdater:
    """Start the background discovery updater thread.

    Args:
        redis_host: Redis server hostname.
        redis_port: Redis server port.
        redis_password: Optional Redis password.
        update_interval: Seconds between updates.

    Returns:
        The MetricsDiscoveryUpdater thread.
    """
    global _discovery_updater

    if _discovery_updater is not None and _discovery_updater.is_alive():
        return _discovery_updater

    _discovery_updater = MetricsDiscoveryUpdater(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_password=redis_password,
        update_interval=update_interval,
    )
    _discovery_updater.start()
    return _discovery_updater
