"""Prometheus metrics for distributed training.

This module provides Prometheus metrics that can be scraped by the existing
Prometheus instance started by Ray. Metrics are exposed on a separate HTTP
port from each worker.

Workers register their metrics endpoints in Redis for dynamic discovery.
A background thread updates the Prometheus service discovery file.

System metrics (CPU, RAM, GPU, GPU RAM) are collected automatically via a
background thread using psutil and pynvml.

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
from typing import Optional, Dict, Any, List
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    CollectorRegistry,
    start_http_server,
    REGISTRY,
)

# Optional imports for system metrics
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

# TPU monitoring via libtpu SDK (available when jax[tpu] is installed)
try:
    from libtpu.sdk import tpumonitoring
    HAS_TPU_MONITORING = True
except ImportError:
    HAS_TPU_MONITORING = False


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

        # =================================================================
        # Per-Outcome Value Loss Metrics (6-way value head)
        # =================================================================
        self.value_loss_per_outcome = Gauge(
            'bgai_value_loss_per_outcome',
            'Value loss contribution per outcome type',
            ['worker_id', 'outcome'],
            registry=self.registry,
        )

        self.predicted_outcome_prob = Gauge(
            'bgai_predicted_outcome_prob',
            'Mean predicted probability per outcome type',
            ['worker_id', 'outcome'],
            registry=self.registry,
        )

        self.target_outcome_prob = Gauge(
            'bgai_target_outcome_prob',
            'Mean target probability per outcome type (ground truth distribution)',
            ['worker_id', 'outcome'],
            registry=self.registry,
        )

        self.value_accuracy = Gauge(
            'bgai_value_accuracy',
            'Top-1 accuracy of value predictions (predicted outcome matches target)',
            ['worker_id'],
            registry=self.registry,
        )

        self.policy_accuracy = Gauge(
            'bgai_policy_accuracy',
            'Top-1 accuracy of policy predictions',
            ['worker_id'],
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

        self.mcts_temperature = Gauge(
            'bgai_mcts_temperature',
            'Current MCTS exploration temperature (decays over game progress)',
            ['worker_id'],
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

        # =================================================================
        # System Metrics (CPU, RAM, GPU, GPU RAM)
        # =================================================================
        # System metrics use 'host' label so multiple workers on same host
        # update the same time series (avoiding duplication)
        self.cpu_percent = Gauge(
            'bgai_cpu_percent',
            'CPU utilization percentage (0-100)',
            ['host'],
            registry=self.registry,
        )

        self.cpu_count = Gauge(
            'bgai_cpu_count',
            'Number of CPU cores',
            ['host'],
            registry=self.registry,
        )

        self.memory_used_bytes = Gauge(
            'bgai_memory_used_bytes',
            'System RAM used in bytes',
            ['host'],
            registry=self.registry,
        )

        self.memory_total_bytes = Gauge(
            'bgai_memory_total_bytes',
            'Total system RAM in bytes',
            ['host'],
            registry=self.registry,
        )

        self.memory_percent = Gauge(
            'bgai_memory_percent',
            'System RAM utilization percentage (0-100)',
            ['host'],
            registry=self.registry,
        )

        self.gpu_utilization_percent = Gauge(
            'bgai_gpu_utilization_percent',
            'GPU compute utilization percentage (0-100)',
            ['host', 'gpu_index', 'gpu_name'],
            registry=self.registry,
        )

        self.gpu_memory_used_bytes = Gauge(
            'bgai_gpu_memory_used_bytes',
            'GPU memory used in bytes',
            ['host', 'gpu_index', 'gpu_name'],
            registry=self.registry,
        )

        self.gpu_memory_total_bytes = Gauge(
            'bgai_gpu_memory_total_bytes',
            'Total GPU memory in bytes',
            ['host', 'gpu_index', 'gpu_name'],
            registry=self.registry,
        )

        self.gpu_memory_percent = Gauge(
            'bgai_gpu_memory_percent',
            'GPU memory utilization percentage (0-100)',
            ['host', 'gpu_index', 'gpu_name'],
            registry=self.registry,
        )

        self.gpu_temperature = Gauge(
            'bgai_gpu_temperature_celsius',
            'GPU temperature in Celsius',
            ['host', 'gpu_index', 'gpu_name'],
            registry=self.registry,
        )

        self.gpu_power_watts = Gauge(
            'bgai_gpu_power_watts',
            'GPU power draw in watts',
            ['host', 'gpu_index', 'gpu_name'],
            registry=self.registry,
        )

        # =================================================================
        # JAX Memory Metrics (per-worker, requires cuda_async allocator)
        # These track actual JAX memory usage, not just GPU allocation
        # =================================================================
        self.jax_memory_bytes_in_use = Gauge(
            'bgai_jax_memory_bytes_in_use',
            'JAX memory currently in use (bytes)',
            ['worker_id'],
            registry=self.registry,
        )

        self.jax_memory_peak_bytes = Gauge(
            'bgai_jax_memory_peak_bytes',
            'JAX peak memory usage (bytes)',
            ['worker_id'],
            registry=self.registry,
        )

        self.jax_memory_bytes_limit = Gauge(
            'bgai_jax_memory_bytes_limit',
            'JAX memory limit (bytes)',
            ['worker_id'],
            registry=self.registry,
        )

        self.jax_memory_num_allocs = Gauge(
            'bgai_jax_memory_num_allocs',
            'Number of JAX memory allocations',
            ['worker_id'],
            registry=self.registry,
        )

        # =================================================================
        # TPU Metrics (via libtpu SDK)
        # =================================================================
        self.tpu_duty_cycle_percent = Gauge(
            'bgai_tpu_duty_cycle_percent',
            'TPU TensorCore duty cycle percentage (0-100)',
            ['host', 'tpu_index'],
            registry=self.registry,
        )

        self.tpu_tensor_core_utilization = Gauge(
            'bgai_tpu_tensor_core_utilization',
            'TPU Tensor Core utilization percentage (0-100)',
            ['host', 'tpu_index'],
            registry=self.registry,
        )

        self.tpu_hbm_memory_used_bytes = Gauge(
            'bgai_tpu_hbm_memory_used_bytes',
            'TPU High Bandwidth Memory used in bytes',
            ['host', 'tpu_index'],
            registry=self.registry,
        )

        self.tpu_hbm_memory_total_bytes = Gauge(
            'bgai_tpu_hbm_memory_total_bytes',
            'TPU High Bandwidth Memory total in bytes',
            ['host', 'tpu_index'],
            registry=self.registry,
        )

        self.tpu_hbm_memory_percent = Gauge(
            'bgai_tpu_hbm_memory_percent',
            'TPU High Bandwidth Memory utilization percentage (0-100)',
            ['host', 'tpu_index'],
            registry=self.registry,
        )

        # =================================================================
        # Worker Task Status
        # =================================================================
        self.current_task = Info(
            'bgai_current_task',
            'Current task being performed by the worker',
            ['worker_id'],
            registry=self.registry,
        )

        self.task_iterations = Counter(
            'bgai_task_iterations_total',
            'Total iterations/steps of current task',
            ['worker_id', 'task_type'],
            registry=self.registry,
        )

        self.task_duration_seconds = Gauge(
            'bgai_task_duration_seconds',
            'Time spent on current task in seconds',
            ['worker_id', 'task_type'],
            registry=self.registry,
        )


# Global metrics instance
_metrics: Optional[BGAIMetrics] = None
_metrics_lock = threading.Lock()
_server_started = False
_server_port: Optional[int] = None


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


def start_metrics_server(port: int = 9100) -> Optional[int]:
    """Start the Prometheus metrics HTTP server.

    Args:
        port: Port to serve metrics on.

    Returns:
        The actual port the server is running on, or None if failed to start.
        Returns the existing port if server was already running.
    """
    global _server_started, _server_port
    with _metrics_lock:
        if _server_started:
            return _server_port

        try:
            start_http_server(port)
            _server_started = True
            _server_port = port
            print(f"Prometheus metrics server started on port {port}")
            return port
        except Exception as e:
            print(f"Failed to start metrics server on port {port}: {e}")
            # Try alternative ports
            for alt_port in range(port + 1, port + 10):
                try:
                    start_http_server(alt_port)
                    _server_started = True
                    _server_port = alt_port
                    print(f"Prometheus metrics server started on port {alt_port}")
                    return alt_port
                except Exception:
                    continue
            return None


def reset_metrics():
    """Reset the global metrics instance (for testing)."""
    global _metrics, _server_started, _server_port
    with _metrics_lock:
        _metrics = None
        _server_started = False
        _server_port = None


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

    # Group by worker type (using sets to deduplicate endpoints)
    game_targets = set()
    training_targets = set()
    eval_targets = set()

    for worker_id, info in workers.items():
        endpoint = info.get("endpoint", "")
        worker_type = info.get("worker_type", "unknown")

        if worker_type == "game":
            game_targets.add(endpoint)
        elif worker_type == "training":
            training_targets.add(endpoint)
        elif worker_type == "eval":
            eval_targets.add(endpoint)

    # Build Prometheus file_sd_configs format
    discovery = []

    if game_targets:
        discovery.append({
            "labels": {"job": "bgai_game_workers", "worker_type": "game"},
            "targets": sorted(game_targets),
        })

    if training_targets:
        discovery.append({
            "labels": {"job": "bgai_training_workers", "worker_type": "training"},
            "targets": sorted(training_targets),
        })

    if eval_targets:
        discovery.append({
            "labels": {"job": "bgai_eval_workers", "worker_type": "eval"},
            "targets": sorted(eval_targets),
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


# =============================================================================
# System Metrics Collector
# =============================================================================

class SystemMetricsCollector(threading.Thread):
    """Background thread that collects system metrics (CPU, RAM, GPU).

    Collects metrics every `update_interval` seconds and updates Prometheus gauges.
    Works on both NVIDIA GPUs (via pynvml) and non-GPU systems (via psutil).
    """

    def __init__(
        self,
        worker_id: str,
        update_interval: float = 5.0,
        gpu_indices: Optional[List[int]] = None,
    ):
        """Initialize the system metrics collector.

        Args:
            worker_id: Unique worker identifier (used for logging, not metrics).
            update_interval: Seconds between metric collections.
            gpu_indices: List of GPU indices to monitor (None = all available).
        """
        super().__init__(daemon=True)
        self.worker_id = worker_id
        # Use hostname for system metrics to avoid duplication when multiple
        # workers run on the same host
        self.hostname = socket.gethostname()
        self.update_interval = update_interval
        self.gpu_indices = gpu_indices
        self._stop_event = threading.Event()
        self._nvml_initialized = False
        self._gpu_handles: List[Any] = []
        self._gpu_names: List[str] = []

    def _init_nvml(self) -> bool:
        """Initialize NVML for GPU monitoring."""
        if not HAS_PYNVML:
            return False

        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True

            # Get GPU handles
            device_count = pynvml.nvmlDeviceGetCount()
            indices = self.gpu_indices if self.gpu_indices else list(range(device_count))

            for idx in indices:
                if idx < device_count:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    self._gpu_handles.append((idx, handle))
                    self._gpu_names.append(name)

            return True
        except Exception as e:
            print(f"Failed to initialize NVML: {e}")
            return False

    def _shutdown_nvml(self):
        """Shutdown NVML."""
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False

    def _collect_cpu_metrics(self, metrics: BGAIMetrics):
        """Collect CPU and memory metrics."""
        if not HAS_PSUTIL:
            return

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()

            metrics.cpu_percent.labels(host=self.hostname).set(cpu_percent)
            metrics.cpu_count.labels(host=self.hostname).set(cpu_count)

            # Memory metrics
            mem = psutil.virtual_memory()
            metrics.memory_used_bytes.labels(host=self.hostname).set(mem.used)
            metrics.memory_total_bytes.labels(host=self.hostname).set(mem.total)
            metrics.memory_percent.labels(host=self.hostname).set(mem.percent)

        except Exception as e:
            print(f"Error collecting CPU metrics: {e}")

    def _collect_gpu_metrics(self, metrics: BGAIMetrics):
        """Collect GPU metrics via NVML."""
        if not self._nvml_initialized:
            return

        try:
            for (idx, handle), name in zip(self._gpu_handles, self._gpu_names):
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics.gpu_utilization_percent.labels(
                    host=self.hostname,
                    gpu_index=str(idx),
                    gpu_name=name,
                ).set(util.gpu)

                # GPU memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics.gpu_memory_used_bytes.labels(
                    host=self.hostname,
                    gpu_index=str(idx),
                    gpu_name=name,
                ).set(mem_info.used)
                metrics.gpu_memory_total_bytes.labels(
                    host=self.hostname,
                    gpu_index=str(idx),
                    gpu_name=name,
                ).set(mem_info.total)
                metrics.gpu_memory_percent.labels(
                    host=self.hostname,
                    gpu_index=str(idx),
                    gpu_name=name,
                ).set(100.0 * mem_info.used / mem_info.total if mem_info.total > 0 else 0)

                # GPU temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    metrics.gpu_temperature.labels(
                        host=self.hostname,
                        gpu_index=str(idx),
                        gpu_name=name,
                    ).set(temp)
                except Exception:
                    pass

                # GPU power
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                    metrics.gpu_power_watts.labels(
                        host=self.hostname,
                        gpu_index=str(idx),
                        gpu_name=name,
                    ).set(power)
                except Exception:
                    pass

        except Exception as e:
            print(f"Error collecting GPU metrics: {e}")

    def _collect_tpu_metrics(self, metrics: BGAIMetrics):
        """Collect TPU metrics via libtpu SDK.

        Uses the libtpu.sdk.tpumonitoring module to collect TPU metrics
        including duty cycle, tensor core utilization, and HBM memory usage.
        """
        if not HAS_TPU_MONITORING:
            return

        try:
            # Get available metrics from libtpu
            # The SDK provides metrics like duty_cycle_pct, tensor_core_util_pct,
            # hbm_capacity_total, hbm_capacity_usage, etc.
            supported = tpumonitoring.list_supported_metrics()

            # Collect metrics for each TPU chip
            # The actual metric names may vary by TPU generation
            for tpu_idx in range(8):  # Max 8 TPU cores per host typically
                try:
                    # Duty cycle percentage
                    if 'duty_cycle_pct' in supported:
                        duty_cycle = tpumonitoring.get_metric('duty_cycle_pct', chip=tpu_idx)
                        if duty_cycle is not None:
                            metrics.tpu_duty_cycle_percent.labels(
                                host=self.hostname,
                                tpu_index=str(tpu_idx),
                            ).set(duty_cycle)

                    # Tensor core utilization
                    if 'tensor_core_util_pct' in supported:
                        tc_util = tpumonitoring.get_metric('tensor_core_util_pct', chip=tpu_idx)
                        if tc_util is not None:
                            metrics.tpu_tensor_core_utilization.labels(
                                host=self.hostname,
                                tpu_index=str(tpu_idx),
                            ).set(tc_util)

                    # HBM memory
                    if 'hbm_capacity_total' in supported and 'hbm_capacity_usage' in supported:
                        hbm_total = tpumonitoring.get_metric('hbm_capacity_total', chip=tpu_idx)
                        hbm_used = tpumonitoring.get_metric('hbm_capacity_usage', chip=tpu_idx)
                        if hbm_total is not None and hbm_used is not None:
                            metrics.tpu_hbm_memory_total_bytes.labels(
                                host=self.hostname,
                                tpu_index=str(tpu_idx),
                            ).set(hbm_total)
                            metrics.tpu_hbm_memory_used_bytes.labels(
                                host=self.hostname,
                                tpu_index=str(tpu_idx),
                            ).set(hbm_used)
                            if hbm_total > 0:
                                metrics.tpu_hbm_memory_percent.labels(
                                    host=self.hostname,
                                    tpu_index=str(tpu_idx),
                                ).set(100.0 * hbm_used / hbm_total)

                except Exception:
                    # No more TPU chips or metric not available for this chip
                    break

        except Exception as e:
            # Only log once to avoid spam
            if not hasattr(self, '_tpu_error_logged'):
                print(f"Error collecting TPU metrics: {e}")
                self._tpu_error_logged = True

    def _collect_jax_memory_metrics(self, metrics: BGAIMetrics):
        """Collect JAX memory metrics via device.memory_stats().

        Requires XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async to get meaningful stats.
        With the default BFC allocator, bytes_in_use will always be 0.
        """
        try:
            import jax
            devices = jax.devices()
            if not devices:
                return

            # Get memory stats from primary device
            device = devices[0]
            stats = device.memory_stats()

            if stats:
                metrics.jax_memory_bytes_in_use.labels(
                    worker_id=self.worker_id
                ).set(stats.get('bytes_in_use', 0))

                metrics.jax_memory_peak_bytes.labels(
                    worker_id=self.worker_id
                ).set(stats.get('peak_bytes_in_use', 0))

                metrics.jax_memory_bytes_limit.labels(
                    worker_id=self.worker_id
                ).set(stats.get('bytes_limit', 0))

                metrics.jax_memory_num_allocs.labels(
                    worker_id=self.worker_id
                ).set(stats.get('num_allocs', 0))

        except Exception as e:
            # Only log once to avoid spam
            if not hasattr(self, '_jax_error_logged'):
                print(f"Error collecting JAX memory metrics: {e}")
                self._jax_error_logged = True

    def run(self):
        """Main collection loop."""
        # Initialize NVML if available
        self._init_nvml()

        # Initialize CPU percent (first call always returns 0)
        if HAS_PSUTIL:
            psutil.cpu_percent(interval=None)

        # Log TPU availability
        if HAS_TPU_MONITORING:
            print(f"TPU monitoring available for worker {self.worker_id}")

        metrics = get_metrics()

        while not self._stop_event.is_set():
            try:
                self._collect_cpu_metrics(metrics)
                self._collect_gpu_metrics(metrics)
                self._collect_jax_memory_metrics(metrics)
                self._collect_tpu_metrics(metrics)
            except Exception as e:
                print(f"Error in system metrics collection: {e}")

            self._stop_event.wait(self.update_interval)

        self._shutdown_nvml()

    def stop(self):
        """Stop the collector thread."""
        self._stop_event.set()


# Global system metrics collector instance
_system_metrics_collector: Optional[SystemMetricsCollector] = None


def start_system_metrics_collector(
    worker_id: str,
    update_interval: float = 5.0,
    gpu_indices: Optional[List[int]] = None,
) -> Optional[SystemMetricsCollector]:
    """Start the system metrics collector thread.

    Args:
        worker_id: Unique worker identifier for metric labels.
        update_interval: Seconds between metric collections.
        gpu_indices: List of GPU indices to monitor (None = all available).

    Returns:
        The SystemMetricsCollector thread, or None if already running.
    """
    global _system_metrics_collector

    if _system_metrics_collector is not None and _system_metrics_collector.is_alive():
        return None

    _system_metrics_collector = SystemMetricsCollector(
        worker_id=worker_id,
        update_interval=update_interval,
        gpu_indices=gpu_indices,
    )
    _system_metrics_collector.start()
    print(f"System metrics collector started for worker {worker_id}")
    return _system_metrics_collector


def stop_system_metrics_collector():
    """Stop the system metrics collector thread."""
    global _system_metrics_collector
    if _system_metrics_collector is not None:
        _system_metrics_collector.stop()
        _system_metrics_collector = None
