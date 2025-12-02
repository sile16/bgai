"""Base worker class for distributed training.

This module provides the abstract base class for all worker types
(game, training, evaluation) with common functionality like:
- Coordinator registration and heartbeat
- Device detection and configuration
- Graceful shutdown handling
"""

import time
import threading
import socket
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any

import ray

from ..device import detect_device, get_device_config, DeviceInfo
from ..config import DistributedConfig


@dataclass
class WorkerStats:
    """Statistics tracked by workers."""
    games_generated: int = 0
    training_steps: int = 0
    episodes_completed: int = 0
    experiences_collected: int = 0
    start_time: float = 0.0
    last_heartbeat_time: float = 0.0

    def to_heartbeat_stats(self, reset: bool = True) -> Dict[str, Any]:
        """Convert to stats dict for heartbeat, optionally resetting counters."""
        stats = {
            'games_since_last': self.games_generated,
            'training_steps_since_last': self.training_steps,
            'status': 'working',
        }
        if reset:
            self.games_generated = 0
            self.training_steps = 0
        return stats


class BaseWorker(ABC):
    """Abstract base class for distributed workers.

    Provides common functionality for all worker types:
    - Coordinator communication (registration, heartbeat)
    - Device detection and configuration
    - Graceful shutdown handling

    Subclasses must implement:
    - _run_loop(): Main worker loop
    - worker_type: Property returning worker type string
    """

    def __init__(
        self,
        coordinator_handle: ray.actor.ActorHandle,
        worker_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the base worker.

        Args:
            coordinator_handle: Ray actor handle for the coordinator.
            worker_id: Optional unique worker ID. Auto-generated if not provided.
            config: Optional configuration dict from coordinator.
        """
        self.coordinator = coordinator_handle
        self.worker_id = worker_id or self._generate_worker_id()
        self.config = config or {}

        # Device info
        self.device_info = detect_device()
        self.device_config = get_device_config(self.device_info)

        # State
        self.running = False
        self.registered = False
        self.current_model_version = 0

        # Statistics
        self.stats = WorkerStats(start_time=time.time())

        # Heartbeat thread
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_interval = self.config.get('heartbeat_interval', 10.0)

    def _generate_worker_id(self) -> str:
        """Generate a unique worker ID."""
        hostname = socket.gethostname()
        short_uuid = uuid.uuid4().hex[:8]
        return f"{self.worker_type}-{hostname}-{short_uuid}"

    @property
    @abstractmethod
    def worker_type(self) -> str:
        """Return the worker type string ('game', 'training', 'evaluation')."""
        pass

    # =========================================================================
    # Coordinator Communication
    # =========================================================================

    def register(self) -> bool:
        """Register with the coordinator.

        Returns:
            True if registration was successful.
        """
        device_info_dict = {
            'device_type': 'cuda' if self.device_info.is_cuda else 'metal' if self.device_info.is_metal else 'cpu',
            'device_name': self.device_info.device_kind,
            'hostname': socket.gethostname(),
        }

        print(f"Worker {self.worker_id}: Attempting to register with coordinator...")
        print(f"  Device info: {device_info_dict}")

        try:
            print(f"Worker {self.worker_id}: Calling coordinator.register_worker.remote()...")
            future = self.coordinator.register_worker.remote(
                self.worker_id,
                self.worker_type,
                device_info_dict,
            )
            print(f"Worker {self.worker_id}: Waiting for registration response (timeout=30s)...")
            result = ray.get(future, timeout=30)
            print(f"Worker {self.worker_id}: Got registration response: {result}")

            if result['status'] == 'registered':
                self.registered = True
                self.current_model_version = result.get('model_version', 0)

                # Merge coordinator config with our config
                if 'config' in result:
                    self.config.update(result['config'])

                print(f"Worker {self.worker_id} registered successfully (model_version={self.current_model_version})")
                return True
            else:
                print(f"Worker {self.worker_id}: Registration failed: {result}")
                return False

        except ray.exceptions.GetTimeoutError:
            print(f"Worker {self.worker_id}: Registration TIMEOUT - coordinator not responding")
            return False
        except Exception as e:
            import traceback
            print(f"Worker {self.worker_id}: Registration error: {type(e).__name__}: {e}")
            traceback.print_exc()
            return False

    def deregister(self) -> bool:
        """Deregister from the coordinator.

        Returns:
            True if deregistration was successful.
        """
        if not self.registered:
            return True

        try:
            result = ray.get(self.coordinator.deregister_worker.remote(self.worker_id))
            self.registered = False
            print(f"Worker {self.worker_id} deregistered")
            return result['status'] == 'deregistered'
        except Exception as e:
            print(f"Deregistration error: {e}")
            return False

    def _send_heartbeat(self) -> Dict[str, Any]:
        """Send heartbeat to coordinator.

        Returns:
            Coordinator response dict.
        """
        stats = self.stats.to_heartbeat_stats(reset=True)
        self.stats.last_heartbeat_time = time.time()

        try:
            return ray.get(self.coordinator.heartbeat.remote(self.worker_id, stats))
        except Exception as e:
            print(f"Heartbeat error: {e}")
            return {'status': 'error', 'message': str(e)}

    def _heartbeat_loop(self) -> None:
        """Background thread for sending heartbeats."""
        while self.running:
            try:
                response = self._send_heartbeat()

                # Check for model updates
                if response.get('new_model_available'):
                    new_version = response.get('new_model_version', 0)
                    if new_version > self.current_model_version:
                        self._on_model_update_available(new_version)

                # Check if we need to re-register
                if response.get('should_register'):
                    self.register()

            except Exception as e:
                print(f"Heartbeat loop error: {e}")

            # Sleep until next heartbeat
            time.sleep(self._heartbeat_interval)

    def _start_heartbeat(self) -> None:
        """Start the heartbeat background thread."""
        if self._heartbeat_thread is not None:
            return

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"heartbeat-{self.worker_id}",
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        """Stop the heartbeat background thread."""
        # Thread will stop when self.running becomes False
        self._heartbeat_thread = None

    def _on_model_update_available(self, new_version: int) -> None:
        """Called when a new model version is available.

        Subclasses can override to fetch and apply new weights.

        Args:
            new_version: The new model version number.
        """
        pass  # Subclasses implement this

    # =========================================================================
    # Model Weights
    # =========================================================================

    def fetch_model_weights(self) -> Optional[bytes]:
        """Fetch current model weights from coordinator.

        Returns:
            Serialized model weights or None if not available.
        """
        try:
            weights, version = ray.get(self.coordinator.get_model_weights.remote())
            if weights is not None:
                self.current_model_version = version
                self._acknowledge_model_version(version)
            return weights
        except Exception as e:
            print(f"Error fetching model weights: {e}")
            return None

    def _acknowledge_model_version(self, version: int) -> None:
        """Acknowledge receipt of model version to coordinator."""
        try:
            ray.get(self.coordinator.acknowledge_model_version.remote(
                self.worker_id,
                version,
            ))
        except Exception as e:
            print(f"Error acknowledging model version: {e}")

    # =========================================================================
    # Main Loop
    # =========================================================================

    @abstractmethod
    def _run_loop(self, num_iterations: int = -1) -> Dict[str, Any]:
        """Main worker loop. Must be implemented by subclasses.

        Args:
            num_iterations: Number of iterations to run (-1 for infinite).

        Returns:
            Dict with results/statistics from the run.
        """
        pass

    def run(self, num_iterations: int = -1) -> Dict[str, Any]:
        """Run the worker.

        Args:
            num_iterations: Number of iterations to run (-1 for infinite).

        Returns:
            Dict with results/statistics from the run.
        """
        # Register with coordinator
        if not self.register():
            return {'status': 'error', 'message': 'Failed to register'}

        self.running = True

        # Start heartbeat thread
        self._start_heartbeat()

        try:
            # Run the main loop
            result = self._run_loop(num_iterations)
            return result
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the worker gracefully."""
        self.running = False
        self._stop_heartbeat()
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
            'running': self.running,
            'registered': self.registered,
            'uptime_seconds': time.time() - self.stats.start_time,
            'total_games': self.stats.games_generated,
            'total_training_steps': self.stats.training_steps,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.worker_id}, type={self.worker_type})"
