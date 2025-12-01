"""Prometheus metrics for distributed training.

This module provides Prometheus metrics that can be scraped by the existing
Prometheus instance started by Ray. Metrics are exposed on a separate HTTP
port from each worker.

Usage:
    from distributed.metrics import get_metrics, start_metrics_server

    # Start metrics server on worker initialization
    start_metrics_server(port=9100)

    # Get metrics instance and record values
    metrics = get_metrics()
    metrics.games_total.labels(worker_id='gpu-0').inc()
    metrics.training_loss.set(0.5)
"""

import threading
from typing import Optional
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
