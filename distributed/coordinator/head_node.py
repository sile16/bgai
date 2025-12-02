"""Coordinator for distributed training.

The Coordinator is a standalone Python process that manages:
- Model weight storage and distribution via Redis
- Worker health monitoring (via Redis TTL)
- Training run lifecycle
- Cluster status reporting

No Ray dependency - uses Redis for all coordination.
"""

import time
import threading
import signal
import sys
from typing import Dict, Optional, Any, List

from .redis_state import (
    RedisStateManager,
    WorkerInfo,
    WorkerStatus,
    RunStatus,
    create_state_manager,
    HEARTBEAT_INTERVAL,
)
from ..serialization import serialize_weights, deserialize_weights


class Coordinator:
    """Central coordinator managing workers, model versions, and cluster state.

    This is a standalone Python class (not a Ray actor) that uses Redis
    for all state management.

    Example usage:
        >>> coordinator = Coordinator(config)
        >>> coordinator.start()  # Blocks, runs status loop
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the coordinator.

        Args:
            config: Configuration dictionary with:
                - redis_host: Redis server host
                - redis_port: Redis server port
                - redis_password: Redis password (optional)
                - status_interval: Seconds between status prints (default: 10)
        """
        self.config = config or {}

        # Create Redis state manager
        self.state = create_state_manager(self.config)

        # Configuration
        self.status_interval = self.config.get('status_interval', 10.0)

        # Runtime state
        self.running = False
        self.start_time = time.time()
        self._status_thread: Optional[threading.Thread] = None

        # Verify Redis connection
        if not self.state.ping():
            raise ConnectionError(
                f"Cannot connect to Redis at "
                f"{self.config.get('redis_host', 'localhost')}:"
                f"{self.config.get('redis_port', 6379)}"
            )

        print(f"Coordinator initialized at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Redis: {self.config.get('redis_host', 'localhost')}:{self.config.get('redis_port', 6379)}")

    # =========================================================================
    # Model Weight Management
    # =========================================================================

    def set_initial_weights(self, weights_bytes: bytes) -> Dict[str, Any]:
        """Set initial model weights (version 1).

        Args:
            weights_bytes: Serialized model weights.

        Returns:
            Response with status.
        """
        success = self.state.set_model_weights(weights_bytes, version=1)
        if success:
            print("Initial model weights set (version 1)")
            return {'status': 'updated', 'version': 1}
        else:
            return {'status': 'failed', 'reason': 'version_conflict'}

    def update_model_weights(
        self,
        weights_bytes: bytes,
        version: int,
    ) -> Dict[str, Any]:
        """Update model weights (called by training worker).

        Args:
            weights_bytes: Serialized model weights.
            version: New model version number.

        Returns:
            Response with status and version.
        """
        success = self.state.set_model_weights(weights_bytes, version)

        if success:
            print(f"Model weights updated to version {version}")
            return {'status': 'updated', 'version': version}
        else:
            current = self.state.get_model_version()
            print(f"Rejected weight update: version {version} <= current {current}")
            return {
                'status': 'rejected',
                'reason': 'stale_version',
                'current_version': current,
            }

    def get_model_weights(self) -> tuple:
        """Get current model weights and version.

        Returns:
            Tuple of (weights_bytes, version).
        """
        weights = self.state.get_model_weights()
        version = self.state.get_model_version()
        return (weights, version)

    # =========================================================================
    # Training Run Management
    # =========================================================================

    def start_training_run(self, run_id: str) -> Dict[str, Any]:
        """Start a new training run.

        Args:
            run_id: MLflow run ID.

        Returns:
            Response with status.
        """
        current_status = self.state.get_run_status()
        if current_status == RunStatus.RUNNING.value:
            return {
                'status': 'error',
                'reason': 'Training already running',
                'current_run': self.state.get_run_id(),
            }

        self.state.set_run_id(run_id)
        self.state.set_run_status(RunStatus.RUNNING)
        print(f"Training run started: {run_id}")

        return {'status': 'started', 'run_id': run_id}

    def pause_training_run(self) -> Dict[str, Any]:
        """Pause current training run."""
        if self.state.get_run_status() != RunStatus.RUNNING.value:
            return {'status': 'error', 'reason': 'No active training run'}

        self.state.set_run_status(RunStatus.PAUSED)
        run_id = self.state.get_run_id()
        print(f"Training run paused: {run_id}")

        return {'status': 'paused', 'run_id': run_id}

    def resume_training_run(self) -> Dict[str, Any]:
        """Resume paused training run."""
        if self.state.get_run_status() != RunStatus.PAUSED.value:
            return {'status': 'error', 'reason': 'No paused training run'}

        self.state.set_run_status(RunStatus.RUNNING)
        run_id = self.state.get_run_id()
        print(f"Training run resumed: {run_id}")

        return {'status': 'resumed', 'run_id': run_id}

    def stop_training_run(self) -> Dict[str, Any]:
        """Stop current training run."""
        run_id = self.state.get_run_id()
        self.state.set_run_status(RunStatus.STOPPED)
        print(f"Training run stopped: {run_id}")

        return {'status': 'stopped', 'run_id': run_id}

    # =========================================================================
    # Status and Monitoring
    # =========================================================================

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status.

        Returns:
            Dict with cluster status information.
        """
        status = self.state.get_cluster_status()
        status['uptime_seconds'] = time.time() - self.start_time
        return status

    def get_worker_counts(self) -> Dict[str, int]:
        """Get count of workers by type."""
        return self.state.get_worker_counts()

    def print_status(self) -> None:
        """Print current cluster status to stdout."""
        status = self.get_cluster_status()
        counts = status['workers']

        print(
            f"Status: model_v{status['model_version']}, "
            f"games: {counts['game']}, "
            f"training: {counts['training']}, "
            f"total_games: {status['total_games_generated']}, "
            f"train_steps: {status['total_training_steps']}"
        )

    # =========================================================================
    # Configuration
    # =========================================================================

    def get_worker_config(
        self,
        worker_type: str,
        device_type: str,
    ) -> Dict[str, Any]:
        """Get configuration for a specific worker type.

        Args:
            worker_type: Type of worker ('game', 'training', 'eval').
            device_type: Worker's device type ('cuda', 'metal', 'cpu').

        Returns:
            Worker configuration dict.
        """
        base_config = {
            'redis_host': self.config.get('redis_host', 'localhost'),
            'redis_port': self.config.get('redis_port', 6379),
            'redis_password': self.config.get('redis_password'),
            'heartbeat_interval': HEARTBEAT_INTERVAL,
        }

        # Add type-specific config
        if worker_type == 'game':
            base_config.update({
                'num_simulations': self.config.get('mcts_simulations', 100),
                'max_nodes': self.config.get('mcts_max_nodes', 400),
                'batch_size': self.config.get('game_batch_size', 16),
                'max_episode_steps': self.config.get('max_episode_steps', 500),
            })
        elif worker_type == 'training':
            base_config.update({
                'train_batch_size': self.config.get('train_batch_size', 128),
                'learning_rate': self.config.get('learning_rate', 3e-4),
                'checkpoint_interval': self.config.get('checkpoint_interval', 1000),
            })
        elif worker_type == 'eval':
            base_config.update({
                'num_simulations': self.config.get('eval_simulations', 200),
                'num_episodes': self.config.get('eval_episodes', 64),
            })

        return base_config

    def get_config(self) -> Dict[str, Any]:
        """Get full coordinator configuration."""
        return self.config.copy()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self, blocking: bool = True) -> None:
        """Start the coordinator.

        Args:
            blocking: If True, blocks and runs status loop. If False, returns immediately.
        """
        self.running = True
        self.start_time = time.time()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        print("Coordinator started")

        if blocking:
            self._run_status_loop()
        else:
            self._status_thread = threading.Thread(target=self._run_status_loop)
            self._status_thread.daemon = True
            self._status_thread.start()

    def stop(self) -> None:
        """Stop the coordinator gracefully."""
        print("\nCoordinator stopping...")
        self.running = False

        if self._status_thread and self._status_thread.is_alive():
            self._status_thread.join(timeout=5)

        # Cleanup stale workers
        cleaned = self.state.cleanup_stale_workers()
        if cleaned > 0:
            print(f"Cleaned up {cleaned} stale workers")

        print("Coordinator stopped")

    def _run_status_loop(self) -> None:
        """Main loop that prints status and cleans up stale workers."""
        while self.running:
            try:
                self.print_status()

                # Cleanup stale workers periodically
                self.state.cleanup_stale_workers()

                # Update Prometheus targets
                self.state.update_metrics_targets()

                # Sleep in small increments to allow quick shutdown
                for _ in range(int(self.status_interval)):
                    if not self.running:
                        break
                    time.sleep(1)

            except Exception as e:
                print(f"Error in status loop: {e}")
                time.sleep(1)

    def _handle_signal(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        self.stop()
        sys.exit(0)


# =============================================================================
# Standalone Functions (for compatibility)
# =============================================================================

def create_coordinator(config: Optional[Dict[str, Any]] = None) -> Coordinator:
    """Create a new coordinator instance.

    Args:
        config: Coordinator configuration.

    Returns:
        Coordinator instance.
    """
    return Coordinator(config)


def run_coordinator(config: Optional[Dict[str, Any]] = None) -> None:
    """Create and run a coordinator (blocking).

    This is the main entry point for running the coordinator as a standalone process.

    Args:
        config: Coordinator configuration.
    """
    coordinator = create_coordinator(config)
    coordinator.start(blocking=True)
