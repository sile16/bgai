"""Coordinator (Ray Head Node) for distributed training.

The Coordinator is a Ray actor that manages:
- Worker registration and health monitoring
- Model weight storage and version tracking
- Configuration distribution to workers
- Cluster status reporting
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from enum import Enum

import ray

from ..serialization import serialize_weights, deserialize_weights
from ..config import DistributedConfig, get_default_config


class WorkerStatus(Enum):
    """Worker status states."""
    IDLE = "idle"
    WORKING = "working"
    DISCONNECTED = "disconnected"


@dataclass
class WorkerInfo:
    """Information about a registered worker."""
    worker_id: str
    worker_type: str  # 'game', 'training', 'evaluation'
    device_type: str  # 'cuda', 'metal', 'cpu'
    device_name: str
    hostname: str
    registered_at: float
    last_heartbeat: float
    current_model_version: int = 0
    games_generated: int = 0
    training_steps: int = 0
    status: WorkerStatus = WorkerStatus.IDLE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'worker_id': self.worker_id,
            'worker_type': self.worker_type,
            'device_type': self.device_type,
            'device_name': self.device_name,
            'hostname': self.hostname,
            'registered_at': self.registered_at,
            'last_heartbeat': self.last_heartbeat,
            'current_model_version': self.current_model_version,
            'games_generated': self.games_generated,
            'training_steps': self.training_steps,
            'status': self.status.value,
        }


@dataclass
class ClusterStatus:
    """Current cluster status."""
    model_version: int
    active_workers: List[Dict[str, Any]]
    disconnected_count: int
    total_games_generated: int
    total_training_steps: int
    buffer_size: int
    timestamp: float


@ray.remote
class Coordinator:
    """Central coordinator managing workers, model versions, and task distribution.

    Example usage:
        >>> ray.init()
        >>> coordinator = Coordinator.options(name="coordinator").remote(config)
        >>> status = ray.get(coordinator.get_cluster_status.remote())
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the coordinator.

        Args:
            config: Configuration dictionary. If None, uses defaults.
        """
        self.config = config or {}

        # Worker registry
        self.workers: Dict[str, WorkerInfo] = {}
        self._workers_lock = threading.Lock()

        # Model weights
        self.current_model_version: int = 0
        self.model_weights: Optional[bytes] = None
        self._weights_lock = threading.Lock()

        # Configuration
        self.heartbeat_timeout = self.config.get('heartbeat_timeout', 30.0)
        self.heartbeat_interval = self.config.get('heartbeat_interval', 10.0)

        # Statistics
        self.start_time = time.time()

        print(f"Coordinator initialized at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # =========================================================================
    # Worker Registration
    # =========================================================================

    def register_worker(
        self,
        worker_id: str,
        worker_type: str,
        device_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Register a new worker with the coordinator.

        Args:
            worker_id: Unique identifier for the worker.
            worker_type: Type of worker ('game', 'training', 'evaluation').
            device_info: Device information dict with:
                - device_type: 'cuda', 'metal', 'cpu'
                - device_name: Human-readable device name
                - hostname: Worker's hostname

        Returns:
            Registration response with:
                - status: 'registered' or 'error'
                - model_version: Current model version
                - config: Worker configuration
        """
        now = time.time()

        worker_info = WorkerInfo(
            worker_id=worker_id,
            worker_type=worker_type,
            device_type=device_info.get('device_type', 'cpu'),
            device_name=device_info.get('device_name', 'unknown'),
            hostname=device_info.get('hostname', 'unknown'),
            registered_at=now,
            last_heartbeat=now,
            current_model_version=0,
            status=WorkerStatus.IDLE,
        )

        with self._workers_lock:
            self.workers[worker_id] = worker_info

        print(f"Worker registered: {worker_id} ({worker_type}, {device_info.get('device_name', 'unknown')})")

        return {
            'status': 'registered',
            'model_version': self.current_model_version,
            'config': self._get_worker_config(worker_type, device_info.get('device_type', 'cpu')),
        }

    def deregister_worker(self, worker_id: str) -> Dict[str, Any]:
        """Gracefully remove a worker.

        Args:
            worker_id: ID of worker to deregister.

        Returns:
            Response with status.
        """
        with self._workers_lock:
            if worker_id in self.workers:
                worker = self.workers.pop(worker_id)
                print(f"Worker deregistered: {worker_id}")
                return {'status': 'deregistered'}

        return {'status': 'not_found'}

    def heartbeat(
        self,
        worker_id: str,
        stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process worker heartbeat and return any pending updates.

        Args:
            worker_id: ID of the worker.
            stats: Worker statistics:
                - games_since_last: Games generated since last heartbeat
                - training_steps_since_last: Training steps since last heartbeat
                - status: Current worker status ('idle', 'working')

        Returns:
            Response with:
                - status: 'ok' or 'unknown_worker'
                - new_model_available: True if newer model is available
                - new_model_version: New model version number
        """
        with self._workers_lock:
            if worker_id not in self.workers:
                return {'status': 'unknown_worker', 'should_register': True}

            worker = self.workers[worker_id]
            worker.last_heartbeat = time.time()

            # Update statistics
            worker.games_generated += stats.get('games_since_last', 0)
            worker.training_steps += stats.get('training_steps_since_last', 0)

            status_str = stats.get('status', 'idle')
            worker.status = WorkerStatus(status_str) if status_str in [s.value for s in WorkerStatus] else WorkerStatus.IDLE

        response = {'status': 'ok'}

        # Check if worker needs model update
        if worker.current_model_version < self.current_model_version:
            response['new_model_available'] = True
            response['new_model_version'] = self.current_model_version

        return response

    # =========================================================================
    # Model Weight Management
    # =========================================================================

    def update_model_weights(
        self,
        weights_bytes: bytes,
        version: int,
    ) -> Dict[str, Any]:
        """Update the current model weights (called by training worker).

        Note: This assumes a single training worker. Multiple training workers
        could race and this method only accepts strictly newer versions to
        prevent rollback.

        Args:
            weights_bytes: Serialized model weights.
            version: New model version number.

        Returns:
            Response with status and version.
        """
        with self._weights_lock:
            if version <= self.current_model_version:
                # Reject stale or duplicate updates
                print(f"Rejected weight update: version {version} <= current {self.current_model_version}")
                return {
                    'status': 'rejected',
                    'reason': 'stale_version',
                    'current_version': self.current_model_version,
                }

            self.model_weights = weights_bytes
            self.current_model_version = version

        print(f"Model weights updated to version {version}")

        return {'status': 'updated', 'version': version}

    def get_model_weights(self) -> tuple:
        """Get current model weights and version.

        Returns:
            Tuple of (weights_bytes, version).
        """
        with self._weights_lock:
            return (self.model_weights, self.current_model_version)

    def set_initial_weights(self, weights_bytes: bytes) -> Dict[str, Any]:
        """Set initial model weights (version 1).

        Args:
            weights_bytes: Serialized model weights.

        Returns:
            Response with status.
        """
        return self.update_model_weights(weights_bytes, version=1)

    def acknowledge_model_version(
        self,
        worker_id: str,
        version: int,
    ) -> Dict[str, Any]:
        """Acknowledge that a worker has received a model version.

        Args:
            worker_id: ID of the worker.
            version: Model version the worker now has.

        Returns:
            Response with status.
        """
        with self._workers_lock:
            if worker_id in self.workers:
                self.workers[worker_id].current_model_version = version
                return {'status': 'acknowledged'}

        return {'status': 'unknown_worker'}

    # =========================================================================
    # Status and Monitoring
    # =========================================================================

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status for monitoring.

        Returns:
            Dict with cluster status information.
        """
        now = time.time()
        active_workers = []
        disconnected_count = 0

        with self._workers_lock:
            for worker_id, info in self.workers.items():
                if now - info.last_heartbeat > self.heartbeat_timeout:
                    info.status = WorkerStatus.DISCONNECTED
                    disconnected_count += 1
                else:
                    active_workers.append({
                        'id': info.worker_id,
                        'type': info.worker_type,
                        'device': info.device_name,
                        'status': info.status.value,
                        'games': info.games_generated,
                        'train_steps': info.training_steps,
                        'model_version': info.current_model_version,
                        'last_heartbeat_ago': now - info.last_heartbeat,
                    })

            total_games = sum(w.games_generated for w in self.workers.values())
            total_steps = sum(w.training_steps for w in self.workers.values())

        return {
            'model_version': self.current_model_version,
            'active_workers': active_workers,
            'disconnected_count': disconnected_count,
            'total_games_generated': total_games,
            'total_training_steps': total_steps,
            'uptime_seconds': now - self.start_time,
            'timestamp': now,
        }

    def get_worker_count(self) -> Dict[str, int]:
        """Get count of workers by type.

        Returns:
            Dict with worker counts by type.
        """
        now = time.time()
        counts = {'game': 0, 'training': 0, 'evaluation': 0, 'disconnected': 0}

        with self._workers_lock:
            for worker in self.workers.values():
                if now - worker.last_heartbeat > self.heartbeat_timeout:
                    counts['disconnected'] += 1
                else:
                    counts[worker.worker_type] = counts.get(worker.worker_type, 0) + 1

        return counts

    def get_workers_by_type(self, worker_type: str) -> List[Dict[str, Any]]:
        """Get list of workers of a specific type.

        Args:
            worker_type: Type of workers to list.

        Returns:
            List of worker info dicts.
        """
        now = time.time()
        workers = []

        with self._workers_lock:
            for worker in self.workers.values():
                if worker.worker_type == worker_type:
                    if now - worker.last_heartbeat <= self.heartbeat_timeout:
                        workers.append(worker.to_dict())

        return workers

    # =========================================================================
    # Configuration
    # =========================================================================

    def _get_worker_config(
        self,
        worker_type: str,
        device_type: str,
    ) -> Dict[str, Any]:
        """Get configuration for a specific worker type.

        Args:
            worker_type: Type of worker.
            device_type: Worker's device type.

        Returns:
            Worker configuration dict.
        """
        base_config = {
            'redis_host': self.config.get('redis_host', 'localhost'),
            'redis_port': self.config.get('redis_port', 6379),
            'heartbeat_interval': self.heartbeat_interval,
        }

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
                'weight_push_interval': self.config.get('weight_push_interval', 10),
            })
        elif worker_type == 'evaluation':
            base_config.update({
                'num_simulations': self.config.get('eval_simulations', 200),
                'num_episodes': self.config.get('eval_episodes', 64),
            })

        return base_config

    def get_config(self) -> Dict[str, Any]:
        """Get full coordinator configuration.

        Returns:
            Configuration dict.
        """
        return self.config.copy()

    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update coordinator configuration.

        Args:
            updates: Configuration updates.

        Returns:
            Response with status.
        """
        self.config.update(updates)
        return {'status': 'updated', 'config': self.config}

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup_disconnected_workers(self) -> int:
        """Remove workers that have been disconnected for too long.

        Returns:
            Number of workers removed.
        """
        now = time.time()
        removed = 0
        to_remove = []

        with self._workers_lock:
            for worker_id, worker in self.workers.items():
                # Remove if disconnected for more than 5 minutes
                if now - worker.last_heartbeat > 300:
                    to_remove.append(worker_id)

            for worker_id in to_remove:
                del self.workers[worker_id]
                removed += 1

        if removed > 0:
            print(f"Cleaned up {removed} disconnected workers")

        return removed

    def shutdown(self) -> Dict[str, Any]:
        """Graceful shutdown of coordinator.

        Returns:
            Response with status.
        """
        print("Coordinator shutting down...")

        with self._workers_lock:
            worker_count = len(self.workers)
            self.workers.clear()

        return {
            'status': 'shutdown',
            'workers_cleared': worker_count,
        }


# =============================================================================
# Utility Functions
# =============================================================================

def get_coordinator(name: str = "coordinator") -> Optional[ray.actor.ActorHandle]:
    """Get existing coordinator actor by name.

    Args:
        name: Name of the coordinator actor.

    Returns:
        Coordinator actor handle or None if not found.
    """
    try:
        return ray.get_actor(name)
    except ValueError:
        return None


def create_coordinator(
    config: Optional[Dict[str, Any]] = None,
    name: str = "coordinator",
) -> ray.actor.ActorHandle:
    """Create a new coordinator actor.

    Args:
        config: Coordinator configuration.
        name: Name for the coordinator actor.

    Returns:
        Coordinator actor handle.
    """
    return Coordinator.options(name=name).remote(config)


def get_or_create_coordinator(
    config: Optional[Dict[str, Any]] = None,
    name: str = "coordinator",
) -> ray.actor.ActorHandle:
    """Get existing coordinator or create a new one.

    Args:
        config: Coordinator configuration (used only if creating new).
        name: Name for the coordinator actor.

    Returns:
        Coordinator actor handle.
    """
    coordinator = get_coordinator(name)
    if coordinator is None:
        coordinator = create_coordinator(config, name)
    return coordinator
