"""Base worker class for distributed training.

This module provides the abstract base class for all worker types
(game, training, evaluation) with common functionality like:
- Redis-based registration and heartbeat
- Device detection and configuration
- Model weight synchronization
- Graceful shutdown handling

No Ray dependency - uses Redis for all coordination.
"""

import time
import threading
import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any

from ..utils import install_shutdown_handler
from ..coordinator.redis_state import (
    RedisStateManager,
    WorkerInfo,
    create_state_manager,
    HEARTBEAT_INTERVAL,
    WORKER_TTL,
)
from ..device import detect_device, get_device_config, DeviceInfo
from ..metrics import start_system_metrics_collector, stop_system_metrics_collector


@dataclass
class WorkerStats:
    """Statistics tracked by workers."""
    games_generated: int = 0
    training_steps: int = 0
    episodes_completed: int = 0
    experiences_collected: int = 0
    start_time: float = 0.0
    last_heartbeat_time: float = 0.0

    def get_heartbeat_stats(self, reset: bool = True) -> Dict[str, Any]:
        """Get stats for heartbeat, optionally resetting counters."""
        stats = {
            'games_generated': self.games_generated,
            'training_steps': self.training_steps,
            'status': 'working',
        }
        if reset:
            self.games_generated = 0
            self.training_steps = 0
        return stats


class BaseWorker(ABC):
    """Abstract base class for distributed workers.

    Provides common functionality for all worker types:
    - Redis-based registration and heartbeat
    - Device detection and configuration
    - Model weight synchronization
    - Graceful shutdown handling

    Subclasses must implement:
    - _run_loop(): Main worker loop
    - worker_type: Property returning worker type string
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        worker_id: Optional[str] = None,
    ):
        """Initialize the base worker.

        Args:
            config: Configuration dict with:
                - redis_host: Redis server host
                - redis_port: Redis server port
                - redis_password: Redis password (optional)
                - heartbeat_interval: Seconds between heartbeats
                - metrics_port: Prometheus metrics port
            worker_id: Optional unique worker ID. Auto-generated if not provided.
        """
        self.config = config or {}
        self.worker_id = worker_id or self._generate_worker_id()

        # Device info
        self.device_info = detect_device()
        self.device_config = get_device_config(self.device_info)

        # Create Redis state manager
        self.state = create_state_manager(self.config)

        # Verify Redis connection
        if not self.state.ping():
            raise ConnectionError(
                f"Cannot connect to Redis at "
                f"{self.config.get('redis_host', 'localhost')}:"
                f"{self.config.get('redis_port', 6379)}"
            )

        # Configuration
        self.heartbeat_interval = self.config.get('heartbeat_interval', HEARTBEAT_INTERVAL)
        # metrics_port is set later by _get_default_metrics_port() after worker_type is available

        # State
        self.running = False
        self.registered = False
        self.current_model_version = 0

        # Statistics
        self.stats = WorkerStats(start_time=time.time())

        # Heartbeat thread
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop_event = threading.Event()

    @property
    @abstractmethod
    def worker_type(self) -> str:
        """Return the worker type string ('game', 'training', 'eval')."""
        pass

    @property
    def metrics_port(self) -> int:
        """Get the metrics port, using worker-type-specific defaults."""
        default_ports = {'game': 9100, 'training': 9200, 'eval': 9300}
        default = default_ports.get(self.worker_type, 9100)
        return self.config.get('metrics_port', default)

    @abstractmethod
    def _run_loop(self, num_iterations: int = -1) -> Dict[str, Any]:
        """Main worker loop to be implemented by subclasses.

        Args:
            num_iterations: Number of iterations to run (-1 for infinite).

        Returns:
            Dict with results/statistics from the run.
        """
        pass

    # =========================================================================
    # Worker ID Generation
    # =========================================================================

    def _generate_worker_id(self) -> str:
        """Generate a unique worker ID."""
        import uuid
        hostname = socket.gethostname().split('.')[0]
        short_id = uuid.uuid4().hex[:8]
        return f"{self.worker_type}-{hostname}-{short_id}"

    # =========================================================================
    # Registration and Heartbeat
    # =========================================================================

    def register(self) -> bool:
        """Register with Redis.

        Returns:
            True if registration was successful.
        """
        print(f"Worker {self.worker_id}: Registering...")

        info = WorkerInfo(
            worker_id=self.worker_id,
            worker_type=self.worker_type,
            device_type='cuda' if self.device_info.is_cuda else 'metal' if self.device_info.is_metal else 'cpu',
            device_name=self.device_info.device_kind,
            hostname=socket.gethostname(),
            metrics_port=self.metrics_port,
            status='idle',
            model_version=0,
        )

        try:
            self.state.register_worker(info, ttl=WORKER_TTL)
            self.registered = True
            self.current_model_version = self.state.get_model_version()

            print(f"Worker {self.worker_id}: Registered successfully")
            print(f"  Device: {info.device_type} ({info.device_name})")
            print(f"  Model version: {self.current_model_version}")
            return True

        except Exception as e:
            print(f"Worker {self.worker_id}: Registration failed: {e}")
            return False

    def deregister(self) -> bool:
        """Deregister from Redis.

        Returns:
            True if deregistration was successful.
        """
        if not self.registered:
            return True

        try:
            self.state.deregister_worker(self.worker_id)
            self.registered = False
            print(f"Worker {self.worker_id}: Deregistered")
            return True
        except Exception as e:
            print(f"Worker {self.worker_id}: Deregistration failed: {e}")
            return False

    def _heartbeat(self) -> bool:
        """Send heartbeat to Redis.

        Returns:
            True if heartbeat was successful.
        """
        try:
            stats = self.stats.get_heartbeat_stats(reset=True)
            stats['model_version'] = self.current_model_version

            success = self.state.heartbeat_worker(
                self.worker_id,
                stats=stats,
                ttl=WORKER_TTL,
            )

            if not success:
                # Worker key expired, re-register
                print(f"Worker {self.worker_id}: Heartbeat failed, re-registering...")
                return self.register()

            self.stats.last_heartbeat_time = time.time()
            return True

        except Exception as e:
            print(f"Worker {self.worker_id}: Heartbeat error: {e}")
            return False

    def _start_heartbeat(self) -> None:
        """Start the heartbeat background thread."""
        self._heartbeat_stop_event.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        """Stop the heartbeat background thread."""
        self._heartbeat_stop_event.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5)

    def _heartbeat_loop(self) -> None:
        """Background heartbeat loop."""
        while not self._heartbeat_stop_event.is_set():
            self._heartbeat()
            # Wait in small increments for responsive shutdown
            for _ in range(int(self.heartbeat_interval)):
                if self._heartbeat_stop_event.is_set():
                    break
                time.sleep(1)

    # =========================================================================
    # Model Weight Synchronization
    # =========================================================================

    def check_for_model_update(self) -> bool:
        """Check if a newer model is available.

        Returns:
            True if newer model is available.
        """
        latest_version = self.state.get_model_version()
        return latest_version > self.current_model_version

    def get_model_weights(self) -> Optional[tuple]:
        """Get model weights if newer version available.

        Returns:
            Tuple of (weights_bytes, version) if newer available, None otherwise.
        """
        result = self.state.get_model_weights_if_newer(self.current_model_version)
        if result:
            weights, version = result
            self.current_model_version = version
            return result
        return None

    def get_current_model_weights(self) -> Optional[tuple]:
        """Get current model weights regardless of version.

        Returns:
            Tuple of (weights_bytes, version) or None if no weights available.
        """
        weights = self.state.get_model_weights()
        version = self.state.get_model_version()
        if weights:
            self.current_model_version = version
            return (weights, version)
        return None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def run(self, num_iterations: int = -1) -> Dict[str, Any]:
        """Run the worker.

        Args:
            num_iterations: Number of iterations to run (-1 for infinite).

        Returns:
            Dict with results/statistics from the run.
        """
        # Setup signal handlers
        install_shutdown_handler(self.stop)

        # Register with Redis
        if not self.register():
            return {'status': 'error', 'message': 'Failed to register'}

        self.running = True

        # Start heartbeat thread
        self._start_heartbeat()

        # Start system metrics collector (CPU, RAM, GPU monitoring)
        start_system_metrics_collector(self.worker_id)

        try:
            # Run the main loop
            result = self._run_loop(num_iterations)
            return result
        except KeyboardInterrupt:
            print(f"\nWorker {self.worker_id}: Interrupted")
            return {'status': 'interrupted'}
        except Exception as e:
            print(f"Worker {self.worker_id}: Error: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the worker gracefully."""
        print(f"Worker {self.worker_id}: Stopping...")
        self.running = False
        self._stop_heartbeat()
        stop_system_metrics_collector()
        self.deregister()

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get current worker statistics.

        Returns:
            Dict with worker statistics.
        """
        return {
            'worker_id': self.worker_id,
            'worker_type': self.worker_type,
            'device_type': self.device_info.platform,
            'device_name': self.device_info.device_kind,
            'model_version': self.current_model_version,
            'uptime': time.time() - self.stats.start_time,
            'games_generated': self.stats.games_generated,
            'training_steps': self.stats.training_steps,
        }

    def is_training_active(self) -> bool:
        """Check if training is currently active.

        Returns:
            True if training run is active.
        """
        return self.state.is_training_active()
